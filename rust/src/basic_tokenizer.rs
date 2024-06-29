use crate::utils::*;

use std::collections::HashMap;
use indexmap::IndexMap;

pub struct BasicTokenizer {
    mints: IndexMap<(usize, usize), usize>,
    vocab: HashMap<usize, Vec<usize>>,
}

impl BasicTokenizer {
    pub fn new() -> BasicTokenizer {
        let mints = IndexMap::new();
        let vocab: HashMap<usize, Vec<usize>> = HashMap::new();
        BasicTokenizer { mints, vocab }
    }

    pub fn train(&mut self, text: &str, vocab_size: usize) -> () {
        if vocab_size < 256 {
            panic!("Vocab size must be at least 256.");
        }
        
        if text.is_empty() {
            println!("Thoth => train]: String empty. Nothing to train on.");
        }

        println!("[Thoth => train]: Training...");
        self.vocab = (0..=255).map(|idx| (idx, vec![idx])).collect();

        let tokens_u8: Vec<u8> = text.as_bytes().to_vec();
        let tokens: Vec<usize> = tokens_u8.iter().map(|&byte| byte as usize).collect();
        let common_tuples: IndexMap<(usize, usize), usize> = count_common_encoded_tuples(tokens.clone());
        let mut sorted_common_tuples: Vec<(usize, (usize, usize))> = common_tuples
            .iter()
            .map(|(k, &v)| (v, *k))
            .collect();
    
        sorted_common_tuples.sort_by(|a, b| b.0.cmp(&a.0));
        let start_idx: usize = 256;
        self.minter(sorted_common_tuples, tokens, start_idx, vocab_size);
        println!("[Thoth => train]: Training complete.");
    }

    pub fn encoder(&mut self, text: &str) -> Vec<usize> {
        if text.is_empty() {
            println!("Thoth => encoder]: String empty. Nothing to encode.");
        }

        println!("[Thoth => encoder]: Encoding...");
        let tokens_u8: Vec<u8> = text.as_bytes().to_vec();
        let mut tokens: Vec<usize> = tokens_u8.iter().map(|&byte| byte as usize).collect();
        
        while tokens.len() >= 2 {
            let common_tuples: IndexMap<(usize, usize), usize> = count_common_encoded_tuples(tokens.clone());
            let mut pair: (usize, usize) = (0, 0);
            let mut freq: &usize = common_tuples.values().next().unwrap();
            for (key, value) in common_tuples.iter() {
                if value > freq { 
                    freq = value;
                    pair = *key;
                }
            }

            if self.mints.get(&pair).is_none() || pair == (0, 0) {
                break;
            }

            let idx_option: Option<&usize> = self.mints.get(&pair);
            let idx_val: usize; 
            if let Some(idx) = idx_option {
                idx_val = *idx;
            } else {
                panic!("Thoth => encoder]: Could not get pair from mints.");
            }
            tokens = merge(pair, tokens.clone(), idx_val);
        }

        println!("[Thoth => encoder]: Encoding Complete...");
        tokens
    }

    pub fn decoder(&mut self, ids: Vec<usize>) -> String {
        if ids.is_empty() {
            panic!("[Thoth => decoder]: No IDs. Nothing to decode.");
        }
        println!("[Thoth => decoder]: Decoding...");
        let text_bytes: Vec<usize> = ids.iter()
            .map(|&idx| self.vocab.get(&idx).unwrap_or(&Vec::new()).clone())            
            .flatten()
            .collect();

        if text_bytes.is_empty() {
            panic!("[Thoth => decoder]: Something went wrong with converting encoding into bytes.");
        }

        let text_bytes_u8: Vec<u8> = text_bytes.iter().map(|&byte| byte as u8).collect();
        let decoded_text = std::str::from_utf8(&text_bytes_u8).map_err(|e| {
            println!("[Thoth => decoder]: Error decoding: {:?}", e);
            e
        });

        let decoded_text_str = match decoded_text {
            Ok(text) => text.to_string(),
            Err(e) => {
                // Handle the error case here. For example, you can return an empty string or a specific error message.
                println!("[Thoth => decoder]: Error converting to string: {:?}", e);
                "".to_string() // Return an empty string or handle the error as needed
            }
        };

        println!("[Thoth => decoder]: Decoding Complete...");
        decoded_text_str
    }

    fn mint_token(&mut self, mut tuples: Vec<(usize, (usize, usize))>, mut ids: Vec<usize>, idx: usize) -> () {
        let most_common_pair_of_ints: (usize, usize) = (tuples.remove(0)).1;
        println!("[Thoth => minter]: Minting {}, {} into a new token {}", most_common_pair_of_ints.0, most_common_pair_of_ints.1, idx);
        let idx_: usize = idx as usize; 
        self.mints.insert(most_common_pair_of_ints, idx_);

        let vec1_option: Option<&Vec<usize>> = self.vocab.get(&most_common_pair_of_ints.0);
        let vec2_option: Option<&Vec<usize>> = self.vocab.get(&most_common_pair_of_ints.1);

        if let (Some(vec1), Some(vec2)) = (vec1_option, vec2_option) {
            let mut concatenated_vec = vec1.clone();
            concatenated_vec.extend(vec2);
            self.vocab.insert(idx, concatenated_vec);
        } else {
            panic!("[Thoth => minter]: One or both vectors are not found in the vocabulary.");
        }
        
        let mut i = 0;
        while i < ids.len() - 1 {
            if usize::from(ids[i]) == most_common_pair_of_ints.0 && usize::from(ids[i + 1]) == most_common_pair_of_ints.1 {
                ids.splice(i..i+2, [idx].iter().cloned());
                if i > 0 {
                    i -= 1;
                }
            } else {
                i += 1;
            }
        }
    }

    fn minter(&mut self, sorted_tuples: Vec<(usize, (usize, usize))>, ids: Vec<usize>, mut start_index: usize, end_index: usize) -> () {
        while start_index < end_index {
            self.mint_token(sorted_tuples.clone(), ids.clone(), start_index);
            start_index += 1;
        }
    }
}
