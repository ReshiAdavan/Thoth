mod basic_tokenizer;
mod utils;

use basic_tokenizer::BasicTokenizer;
use std::fs;

fn main() {
    basic_tokenizer_main();
}

fn basic_tokenizer_main() { 
    let mut tokenizer = BasicTokenizer::new();
    let file_path = "../data/taylorSwift.txt";
    let file_content = fs::read_to_string(file_path).expect("Could not read file");
    let sample_text = "Unicode sample text...";

    println!("\n{}\n", sample_text);

    tokenizer.train(sample_text, 276);

    let encoded = tokenizer.encoder(&file_content);
    assert!(!encoded.is_empty());
    println!("Encoded: {:?}", encoded);

    let decoded = tokenizer.decoder(encoded);
    assert!(!decoded.is_empty());
    println!("Decoded: {}", decoded);
    assert_eq!(file_content, decoded);
}

// fn regex_tokenizer_main() {}
