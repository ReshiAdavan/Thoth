// The Llama-2 Tokenizer uses sentencepiece, which is what is adopted here

use crate::utils::*;
use std::collections::HashMap;
use indexmap::IndexMap;
use std::cmp;

pub struct Llama2Tokenizer {
    vocab: HashMap<usize, char>,
    token_counter: usize,
    // merges: IndexMap<usize, (usize, usize)> PLEASE GET BACK TO THIS
}

impl Llama2Tokenizer {
    pub fn new() -> Llama2Tokenizer {
        let vocab = HashMap::new();
        let token_counter = 0;
        // let merges = IndexMap::new(); PLEASE GET BACK TO THIS
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
            // Count the frequency of each code point
            let code_counts: IndexMap<char, usize> = self.count_codepoint_frequencies(code_points);
            
            // Find the most frequent code point
            let (most_frequent_code_point, _) = code_counts.iter()
                .max_by_key(|&(_, count)| count)
                .unwrap();
        
            // Merge the most frequent code point into a new token
            let new_token = self.generate_new_token();
            
            // Update the vocabulary and the list of code points
            self.vocab.insert(new_token, most_frequent_code_point);
            // code_points = self.merge_items_replace(code_points, most_frequent_pair, new_token); // PLEASE COME BACK TO THIS
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

        while i < code_points.len() { 
            // Start with the longest possible sequence
            // let sequence_length: usize = ;
            while sequence_length > 0 {
                // let sequence: (, ) = ; 
                // if self.vocab.get(sequence) { 
                //     encoded.push(self.vocab.get(sequence));
                //     i += sequence_length;
                //     break
                // } else {
                //     encoded.push(code_points[i]);
                //     i += 1;
                // }
                // sequence_length -= 1; 
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

        // for {
        //     ...
        // } 

        print("[Thoth => decoder]: Decoding Complete...");
    }

    ////////////////////////////////////////////////////////
    /////////////////// HELPER FUNCTIONS ///////////////////
    ////////////////////////////////////////////////////////

    fn count_codepoint_frequencies(code_points: Vec<u32>) -> HashMap<char, usize> {
        let mut counts = HashMap::new();
        for &codepoint in &code_points {
            if let Some(char) = std::char::from_u32(codepoint) {
                *counts.entry(char).or_insert(0) += 1;
            }
        }
        counts
    }

    // fn merge_items_replace(items: Vec<usize>, pair: (usize, usize), new_item: usize) -> Vec<usize> {
    //     let mut merged_items = Vec::new();
    //     let mut i = 0;
    //     while i < items.len() {
    //         if i + 1 < items.len() && (items[i], items[i + 1]) == pair {
    //             merged_items.push(new_item);
    //             i += 2;
    //         } else {
    //             merged_items.push(items[i]);
    //             i += 1;
    //         }
    //     }
    //     merged_items
    // }
    
    fn generate_new_token(&mut self) -> usize {
        let new_token = self.token_counter;
        self.token_counter += 1;
        new_token
    }

    fn most_common_codepoints(codepoints: Vec<usize>, top_n: usize) -> HashMap<usize, char> {
        let mut counts = HashMap::new();
        for &codepoint in &codepoints {
            *counts.entry(codepoint).or_insert(0) += 1;
        }
        
        let mut sorted_counts: Vec<_> = counts.iter().collect();
        sorted_counts.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let top_codepoints = sorted_counts.into_iter().take(top_n);
        top_codepoints.map(|(&codepoint, _)| (codepoint, std::char::from_u32(codepoint).unwrap()))
            .collect()
    }
}
