//! Train weighted independent byte sequences with incremental pair counts.
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::io::{BufRead, BufReader};
type Pair = (usize, usize);
fn pairs(w: &[usize]) -> HashMap<Pair, i64> {
    let mut c = HashMap::new();
    for p in w.windows(2) {
        *c.entry((p[0], p[1])).or_insert(0) += 1;
    }
    c
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() < 3 {
        return Err("Usage: train_bpe INPUT.jsonl OUTPUT.json [VOCAB_SIZE]".into());
    }
    let size: usize = args.get(3).map(|s| s.parse()).transpose()?.unwrap_or(16384);
    if size < 256 {
        return Err("Vocabulary must be at least 256".into());
    }
    let mut words = Vec::new();
    let mut weights = Vec::new();
    for line in BufReader::new(std::fs::File::open(&args[1])?).lines() {
        let (text, weight): (String, i64) = serde_json::from_str(&line?)?;
        words.push(text.bytes().map(usize::from).collect::<Vec<_>>());
        weights.push(weight);
    }
    let mut counts: HashMap<Pair, i64> = HashMap::new();
    let mut locations: HashMap<Pair, HashSet<usize>> = HashMap::new();
    for (i, w) in words.iter().enumerate() {
        for (p, n) in pairs(w) {
            *counts.entry(p).or_default() += n * weights[i];
            locations.entry(p).or_default().insert(i);
        }
    }
    let mut heap: BinaryHeap<_> = counts.iter().map(|(&p, &n)| (n, Reverse(p))).collect();
    let mut merges = Vec::new();
    for id in 256..size {
        let pair = loop {
            match heap.pop() {
                Some((n, Reverse(p))) if n > 0 && counts.get(&p) == Some(&n) => break p,
                Some(_) => continue,
                None => {
                    std::fs::write(
                        &args[2],
                        serde_json::to_vec(
                            &serde_json::json!({"version":1,"kind":"basic","merges":merges}),
                        )?,
                    )?;
                    return Ok(());
                }
            }
        };
        merges.push(pair);
        let affected: Vec<_> = locations[&pair].iter().copied().collect();
        let mut changed = HashSet::new();
        for i in affected {
            let old = pairs(&words[i]);
            let mut new_word = Vec::new();
            let mut j = 0;
            while j < words[i].len() {
                if j + 1 < words[i].len() && (words[i][j], words[i][j + 1]) == pair {
                    new_word.push(id);
                    j += 2;
                } else {
                    new_word.push(words[i][j]);
                    j += 1;
                }
            }
            let new = pairs(&new_word);
            words[i] = new_word;
            let all: HashSet<_> = old.keys().chain(new.keys()).copied().collect();
            for p in all {
                let delta = (new.get(&p).unwrap_or(&0) - old.get(&p).unwrap_or(&0)) * weights[i];
                *counts.entry(p).or_default() += delta;
                changed.insert(p);
                if new.contains_key(&p) {
                    locations.entry(p).or_default().insert(i);
                } else {
                    locations.entry(p).or_default().remove(&i);
                }
            }
        }
        for p in changed {
            let n = counts[&p];
            if n > 0 {
                heap.push((n, Reverse(p)));
            }
        }
        if heap.len() > 100000.max(counts.len() * 4) {
            heap = counts
                .iter()
                .filter(|(_, n)| **n > 0)
                .map(|(&p, &n)| (n, Reverse(p)))
                .collect();
        }
        if id % 1000 == 0 {
            eprintln!("Basic vocabulary {id}");
        }
    }
    std::fs::write(
        &args[2],
        serde_json::to_vec(&serde_json::json!({"version":1,"kind":"basic","merges":merges}))?,
    )?;
    Ok(())
}
