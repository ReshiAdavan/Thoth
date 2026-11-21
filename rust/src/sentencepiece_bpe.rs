//! Ranked BPE with a linked symbol list and lazily invalidated heap entries.
//! Used for ordinary BPE models with complete byte fallback. Other model types
//! and models containing user-defined/unused pieces use the reference runtime.
use sentencepiece_rs::{ModelType, PieceType, SentencePieceProcessor};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

pub struct FastBpe {
    pieces: HashMap<String, (usize, f32)>,
    bytes: [usize; 256],
}
struct Symbol {
    start: usize,
    end: usize,
    prev: Option<usize>,
    next: Option<usize>,
    alive: bool,
}
#[derive(Clone)]
struct Candidate {
    left: usize,
    right: usize,
    left_end: usize,
    right_end: usize,
    score: f32,
}
impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Candidate {}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .total_cmp(&other.score)
            .then_with(|| other.left.cmp(&self.left))
    }
}
impl FastBpe {
    pub fn new(processor: &SentencePieceProcessor) -> Option<Self> {
        let model = processor.model();
        if model.model_type() != ModelType::Bpe
            || model.pieces().iter().any(|p| {
                matches!(p.kind, PieceType::UserDefined | PieceType::Unused) || !p.score.is_finite()
            })
        {
            return None;
        }
        let mut bytes = [0; 256];
        for (byte, slot) in bytes.iter_mut().enumerate() {
            let id = model.try_piece_to_id(&format!("<0x{byte:02X}>"))?;
            if !model.is_byte(id) {
                return None;
            }
            *slot = id;
        }
        let pieces = model
            .pieces()
            .iter()
            .enumerate()
            .filter(|(_, p)| p.kind == PieceType::Normal)
            .map(|(id, p)| {
                (
                    p.piece.clone(),
                    (id, if p.score == 0.0 { 0.0 } else { p.score }),
                )
            })
            .collect();
        Some(Self { pieces, bytes })
    }

    fn offer(&self, text: &str, nodes: &[Symbol], left: usize, heap: &mut BinaryHeap<Candidate>) {
        let Some(right) = nodes[left].next else {
            return;
        };
        if let Some(&(_, score)) = self.pieces.get(&text[nodes[left].start..nodes[right].end]) {
            heap.push(Candidate {
                left,
                right,
                left_end: nodes[left].end,
                right_end: nodes[right].end,
                score,
            });
        }
    }

    pub fn encode_normalized(&self, text: &str) -> Vec<usize> {
        let mut nodes: Vec<Symbol> = text
            .char_indices()
            .enumerate()
            .map(|(i, (start, ch))| Symbol {
                start,
                end: start + ch.len_utf8(),
                prev: i.checked_sub(1),
                next: Some(i + 1),
                alive: true,
            })
            .collect();
        if nodes.is_empty() {
            return Vec::new();
        }
        nodes.last_mut().unwrap().next = None;
        let mut heap = BinaryHeap::new();
        for left in 0..nodes.len() {
            self.offer(text, &nodes, left, &mut heap);
        }
        while let Some(c) = heap.pop() {
            if !nodes[c.left].alive
                || !nodes[c.right].alive
                || nodes[c.left].next != Some(c.right)
                || nodes[c.left].end != c.left_end
                || nodes[c.right].end != c.right_end
            {
                continue;
            }
            nodes[c.left].end = nodes[c.right].end;
            nodes[c.left].next = nodes[c.right].next;
            nodes[c.right].alive = false;
            if let Some(next) = nodes[c.left].next {
                nodes[next].prev = Some(c.left);
            }
            if let Some(prev) = nodes[c.left].prev {
                self.offer(text, &nodes, prev, &mut heap);
            }
            self.offer(text, &nodes, c.left, &mut heap);
        }
        let mut ids = Vec::new();
        for node in nodes.iter().filter(|n| n.alive) {
            let piece = &text[node.start..node.end];
            if let Some(&(id, _)) = self.pieces.get(piece) {
                ids.push(id);
            } else {
                ids.extend(piece.bytes().map(|b| self.bytes[b as usize]));
            }
        }
        ids
    }
}
