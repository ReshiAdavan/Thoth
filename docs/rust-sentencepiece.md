# Rust SentencePiece / Llama tokenizer

The former prototype is now an active Rust module: llama2_tokenizer.rs. It provides loading, saving, training, canonical and checked encoding, decoding, batches, token inspection, and optional BOS/EOS framing. It does not call Python internally.

## Python API

Install the release extension with pip install ./rust.

    from thoth_native import NativeLlama2Tokenizer

    t = NativeLlama2Tokenizer("models/thoth-sentencepiece-en-code-v1.model")
    ids = t.encode("Hello 🦉")
    assert t.decode(ids) == "Hello 🦉"
    canonical = t.encode("Hello", lossless=False, add_bos=True, add_eos=True)

    custom = NativeLlama2Tokenizer.train("hello hello world", 320)
    custom.save("custom.model")
    reloaded = NativeLlama2Tokenizer.from_bytes(custom.to_bytes())
    assert reloaded.encode_batch(["hello", "a▁b"]) == custom.encode_batch(["hello", "a▁b"])

The static train method returns a new tokenizer; it never mutates a loaded model.

## Rust API

    use thoth_native::llama2_tokenizer::Llama2Tokenizer;
    let t = Llama2Tokenizer::load("tokenizer.model")?;
    let canonical_ids = t.encode("Hello")?;
    let checked_ids = t.encode_lossless("a▁b")?;
    assert_eq!(t.decode(&checked_ids)?, "a▁b");
    let trained = Llama2Tokenizer::train("hello hello world", 320)?;
    trained.save("custom.model")?;

Rust encode is canonical. Python encode defaults to checked mode to match the existing Python adapter. Both expose an explicit choice. BOS/EOS are opt-in and checked against model metadata; they do not imply a chat template.

## Runtime design

The [sentencepiece-rs runtime](https://docs.rs/sentencepiece-rs/0.2.2/sentencepiece_rs/) parses standard models and supplies normalization, decoding and fallback model execution. This is an external pure-Rust dependency, not Google's C++ runtime.

Thoth implements an accelerated BPE path for BPE models with all 256 byte pieces and no user-defined or unused pieces. Unicode characters initially form a linked list of spans into the normalized string. Adjacent known pieces enter a max-heap ordered by score, then leftmost position. Each accepted merge updates only neighboring candidate pairs. Entries whose spans or adjacency changed are discarded. This preserves learned rank and leftmost tie behavior without repeatedly scanning the input. Unigram and other configurations use the dependency's runtime.

Checked mode verifies exact recovery. If canonical normalization changes the text, it uses BYTE-type pieces for the original UTF-8 and checks recovery again. Models that cannot preserve the input return an error. This fallback deliberately changes canonical IDs; select canonical mode for an LLM's tokenizer pipeline.

Models are limited to 16 MiB. Invalid IDs return errors; the Python binding rejects booleans, floats, negative values and out-of-range IDs. Model bytes are retained exactly for saving. Native encode/decode/batch/training release the Python GIL; batch execution itself is sequential.

## Native custom training

Thoth's trainer is implemented in Rust, independently of Google's trainer. It learns deterministic Unicode BPE and writes the standard [SentencePiece ModelProto schema](https://github.com/google/sentencepiece/blob/master/src/sentencepiece_model.proto) using prost.

1. Count duplicate independent nonempty lines; convert ordinary spaces to U+2581.
2. Reserve UNKNOWN/BOS/EOS and 256 BYTE pieces, then include the observed alphabet.
3. Count adjacent candidate strings weighted by line frequency, aggregating different parent pairs that form the same string. Select highest frequency, with lexicographic string ties.
4. Replace non-overlapping pairs left to right. Add the unique piece with decreasing merge score.
5. Stop at the target vocabulary or when no new candidates remain. Save identity normalization, no dummy prefix, preserved spaces and byte fallback.

Candidates are limited to 16 Unicode characters. There is no script, digit or whitespace boundary splitting inside a line. Training input is limited to 2 MiB and targets to 259–65,536 pieces; the target must accommodate observed characters too. Empty/NUL-containing corpora are rejected. Encoding NUL is supported.

This bounded reference trainer recounts pairs each iteration. It is intended for experiments, not the existing large-corpus incremental workflow. Its vocabulary and tie rules do not claim to match Google's trainer or recreate Llama's vocabulary. Upstream SentencePiece must load its output and reproduce canonical IDs; differential tests enforce that contract.

## Evidence and limits

Tests compare the saved release model and generated BPE/Unigram models with upstream Python SentencePiece, including normalization, byte fallback, Unicode, framing, serialization and randomized native training. Merge-heavy randomized tests exercise the accelerated path.

The suite does not ship or test Meta's original Llama model artifact. Original-model compatibility requires loading that artifact and validating its BOS/EOS/prompt policy.

The benchmark compares Python's C++ SentencePiece adapter with the pure-Rust runtime using the same model and checked APIs. See benchmarks/sentencepiece-results.json. It is separate from the Python-versus-Rust Basic BPE comparison. No UI changes are included.

The saved model also matched Python IDs and recovered the original text across all 228 held-out files (6,157,482 UTF-8 bytes; 2,455,723 tokens). The result and model checksum are recorded in benchmarks/rust-sentencepiece-heldout.json.
