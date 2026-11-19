mod basic_tokenizer;
use basic_tokenizer::BasicTokenizer;
fn main() -> Result<(),String> {
    let text=std::env::args().nth(1).unwrap_or_else(||"Hello from Thoth 🦉".into());
    let mut tokenizer=BasicTokenizer::new();tokenizer.train(&text,276)?;
    let ids=tokenizer.encoder(&text);println!("{:?}",ids);println!("{}",tokenizer.decoder(ids)?);Ok(())
}
