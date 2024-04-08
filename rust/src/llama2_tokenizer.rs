// The Llama-2 Tokenizer uses sentencepiece, which is what is adopted here

use crate::utils::*;
use std::collections::HashMap;
use indexmap::IndexMap;
use std::cmp;

pub struct Llama2Tokenizer {
    vocab: HashMap<usize, (usize, usize)>,
    token_counter: usize,
    merges: IndexMap<usize, (usize, usize)>
}

impl Llama2Tokenizer {
    pub fn new() -> Llama2Tokenizer {
        let vocab = HashMap::new();
        let token_counter = 0;
        let merges = IndexMap::new();
        Llama2Tokenizer { vocab, token_counter, merges }
    }

    pub fn train(&mut self, text: &str, vocab_size: usize) -> () {
        if vocab_size < 256 {
            panic!("Vocab size must be at least 256.");
        }
        
        if text.is_empty() {
            println!("Thoth => train]: String empty. Nothing to train on.");
        }

        println!("[Thoth => train]: Training...");

        // Convert the text to a list of Unicode code points
        let mut code_points: Vec<u32> = text.chars().map(|c| c as u32).collect();
        self.token_counter = code_points.iter().fold(0.0f32, |max, &val| if val > max { val } else { max });
        
        // Initialize the vocabulary with the most common code points
        self.vocab = self.most_common_codepoints(code_points);

        // Initialize the BPE merges
        let num_merges: u8 = vocab_size - 256;

        for i in 0..num_merges {
            // Count the frequency of each pair of code points
            let pair_counts: IndexMap<(usize, usize), usize> = self.count_pair_frequencies(code_points);
            
            // Find the most frequent pair of code points
            let mut pair: (usize, usize) = (0, 0);
            let mut freq: &usize = pair_counts.values().next().unwrap();
            for (key, value) in common_tuples.iter() {
                if value > freq { 
                    freq = value;
                    pair = *key;
                }
            }

            if pair == (0, 0) {
                panic!("[Thoth => train]: For some reason pair unassigned from before.");
            }

            let most_frequent_pair: (usize, usize) = pair;
            
            // Merge the most frequent pair into a new token
            let new_token = self.generate_new_token();
            
            // Update the vocabulary and the list of code points
            self.vocab.insert(new_token, most_frequent_pair);
            code_points = self.merge_items_replace(code_points, most_frequent_pair, new_token);
        }
        
        print("[Thoth => train]: Training complete.")
    }

    pub fn encoder(&mut self, text: &str) -> Vec<usize> {
        if text.is_empty() {
            println!("Thoth => encoder]: String empty. Nothing to encode.");
        }

        println!("[Thoth => encoder]: Encoding...");
        // Convert the text to a list of Unicode code points
        let mut code_points: Vec<u32> = text.chars().map(|c| c as u32).collect();

        // Encode the text using the trained BPE model
        let encoded: Vec<usize> = Vec::new();

        for i in 0..code_points.len() { 
            // Start with the longest possible sequence
            let sequence_length: usize = ;
            while sequence_length > 0 {
                let sequence: (, ) = ; 
                if self.vocab.get(sequence) { 
                    encoded.push(self.vocab.get(sequence));
                    i += sequence_length;
                    break
                } else {
                    encoded.push(code_points[i]);
                    i += 1;
                }
                sequence_length -= 1; 
            }
        }

        print("[Thoth => encoder]: Encoding Complete...");
        encoded
    }

    pub fn decoder(&mut self, ids: Vec<usize>) -> String {
        if ids.is_empty() {
            panic!("[Thoth => decoder]: No IDs. Nothing to decode.");
        }
        println!("[Thoth => decoder]: Decoding...");

        let decoded: Vec<usize> = Vec::new();

        for {
            ...
        }

        print("[Thoth => decoder]: Decoding Complete...");
    }

    ////////////////////////////////////////////////////////
    /////////////////// HELPER FUNCTIONS ///////////////////
    ////////////////////////////////////////////////////////

    fn count_pair_frequencies(&mut self, code_points: Vec<u32>) -> IndexMap<(usize, usize), usize> {

    }
        

    fn merge_items_replace(&mut self, items: Vec<usize>, pair: (usize, usize), new_item: usize) {

    }
    
    fn generate_new_token(&mut self) {

    }

    fn most_common_codepoints(&mut self, code_points: Vec<u32>, top_n: usize) {

    }
}
