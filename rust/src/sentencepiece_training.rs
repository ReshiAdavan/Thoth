//! Thoth's deterministic Unicode BPE trainer and minimal ModelProto writer.
//! Wire tags follow Google's sentencepiece_model.proto; unspecified fields
//! retain upstream defaults. This is a custom trainer, not Google's trainer.
use prost::Message;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, PartialEq, Message)]
struct Piece {
    #[prost(string, tag = "1")]
    text: String,
    #[prost(float, tag = "2")]
    score: f32,
    #[prost(int32, tag = "3")]
    kind: i32,
}
#[derive(Clone, PartialEq, Message)]
struct Trainer {
    #[prost(int32, tag = "3")]
    model_type: i32,
    #[prost(int32, tag = "4")]
    vocab_size: i32,
    #[prost(bool, optional, tag = "21")]
    split_script: Option<bool>,
    #[prost(bool, optional, tag = "22")]
    split_whitespace: Option<bool>,
    #[prost(bool, optional, tag = "23")]
    split_numbers: Option<bool>,
    #[prost(bool, optional, tag = "33")]
    hard_vocab_limit: Option<bool>,
    #[prost(bool, tag = "35")]
    byte_fallback: bool,
}
#[derive(Clone, PartialEq, Message)]
struct Normalizer {
    #[prost(string, tag = "1")]
    name: String,
    #[prost(bool, optional, tag = "3")]
    add_dummy_prefix: Option<bool>,
    #[prost(bool, optional, tag = "4")]
    remove_extra_whitespaces: Option<bool>,
    #[prost(bool, optional, tag = "5")]
    escape_whitespaces: Option<bool>,
}
#[derive(Clone, PartialEq, Message)]
struct Model {
    #[prost(message, repeated, tag = "1")]
    pieces: Vec<Piece>,
    #[prost(message, optional, tag = "2")]
    trainer: Option<Trainer>,
    #[prost(message, optional, tag = "3")]
    normalizer: Option<Normalizer>,
}

pub fn train(text: &str, size: usize) -> Result<Vec<u8>, String> {
    if text.is_empty() || text.len() > 2 * 1024 * 1024 || !(259..=65536).contains(&size) {
        return Err("Training needs 1 byte–2 MiB of text and vocabulary size 259–65536".into());
    }
    if text.contains('\0') {
        return Err("NUL is not supported in training samples; encoding uses byte fallback".into());
    }
    // Independent lines; spaces use SentencePiece's reversible-in-the-normal-case
    // marker. Literal markers are handled by the checked runtime byte fallback.
    let mut samples: BTreeMap<String, usize> = BTreeMap::new();
    for line in text.lines().filter(|line| !line.is_empty()) {
        *samples.entry(line.replace(' ', "▁")).or_default() += 1;
    }
    if samples.is_empty() {
        return Err("Training needs at least one nonempty line".into());
    }
    let alphabet: BTreeSet<char> = samples.keys().flat_map(|line| line.chars()).collect();
    if size < 259 + alphabet.len() {
        return Err(format!(
            "Vocabulary needs at least {} slots for bytes, special tokens and observed characters",
            259 + alphabet.len()
        ));
    }
    let mut pieces = vec![
        Piece {
            text: "<unk>".into(),
            score: 0.0,
            kind: 2,
        },
        Piece {
            text: "<s>".into(),
            score: 0.0,
            kind: 3,
        },
        Piece {
            text: "</s>".into(),
            score: 0.0,
            kind: 3,
        },
    ];
    for byte in 0..256 {
        pieces.push(Piece {
            text: format!("<0x{byte:02X}>"),
            score: 0.0,
            kind: 6,
        });
    }
    let mut ids: BTreeMap<String, usize> = pieces
        .iter()
        .enumerate()
        .map(|(i, p)| (p.text.clone(), i))
        .collect();
    for ch in alphabet {
        let token = ch.to_string();
        ids.insert(token.clone(), pieces.len());
        pieces.push(Piece {
            text: token,
            score: -(size as f32) - pieces.len() as f32,
            kind: 1,
        });
    }
    let mut words: Vec<(Vec<usize>, usize)> = samples
        .into_iter()
        .map(|(line, weight)| {
            (
                line.chars().map(|ch| ids[&ch.to_string()]).collect(),
                weight,
            )
        })
        .collect();
    let mut rank = 0;
    while pieces.len() < size {
        // Aggregate all ways of constructing the same surface piece; .model
        // vocabularies identify pieces by string, not by a pair of parent IDs.
        let mut counts: BTreeMap<String, usize> = BTreeMap::new();
        for (word, weight) in &words {
            for pair in word.windows(2) {
                let joined = format!("{}{}", pieces[pair[0]].text, pieces[pair[1]].text);
                if joined.chars().count() <= 16 && !ids.contains_key(&joined) {
                    *counts.entry(joined).or_default() += weight;
                }
            }
        }
        let Some((chosen, _)) = counts
            .into_iter()
            .min_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)))
        else {
            break;
        };
        let new_id = pieces.len();
        pieces.push(Piece {
            text: chosen.clone(),
            score: -(rank as f32),
            kind: 1,
        });
        ids.insert(chosen.clone(), new_id);
        rank += 1;
        for (word, _) in &mut words {
            let mut out = Vec::with_capacity(word.len());
            let mut i = 0;
            while i < word.len() {
                if i + 1 < word.len()
                    && format!("{}{}", pieces[word[i]].text, pieces[word[i + 1]].text) == chosen
                {
                    out.push(new_id);
                    i += 2;
                } else {
                    out.push(word[i]);
                    i += 1;
                }
            }
            *word = out;
        }
    }
    let vocab_size = pieces.len() as i32;
    Ok(Model {
        pieces,
        trainer: Some(Trainer {
            model_type: 2,
            vocab_size,
            split_script: Some(false),
            split_whitespace: Some(false),
            split_numbers: Some(false),
            hard_vocab_limit: Some(false),
            byte_fallback: true,
        }),
        normalizer: Some(Normalizer {
            name: "identity".into(),
            add_dummy_prefix: Some(false),
            remove_extra_whitespaces: Some(false),
            escape_whitespaces: Some(true),
        }),
    }
    .encode_to_vec())
}
