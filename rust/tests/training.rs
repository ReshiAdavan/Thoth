#[test]
fn trainer_learns_hierarchical_merges() {
    let dir = std::env::temp_dir().join(format!("thoth-training-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let input = dir.join("input.jsonl");
    let output = dir.join("model.json");
    std::fs::write(&input, "[\"abababab\",3]\n").unwrap();
    let status = std::process::Command::new(env!("CARGO_BIN_EXE_train_bpe"))
        .arg(&input)
        .arg(&output)
        .arg("258")
        .status()
        .unwrap();
    assert!(status.success());
    let model = thoth_native::basic_tokenizer::BasicTokenizer::from_json(
        &std::fs::read_to_string(&output).unwrap(),
    )
    .unwrap();
    assert_eq!(model.encoder("abababab"), vec![257, 257]);
    assert_eq!(
        model.decoder(model.encoder("Hello 🦉")).unwrap(),
        "Hello 🦉"
    );
    std::fs::remove_dir_all(dir).unwrap();
}
