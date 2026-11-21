# Tokenizer contracts and design decisions

## Byte BPE

Basic BPE starts with all 256 byte values. Training counts adjacent pairs including overlaps, selects the largest weighted count, breaks ties by lexicographic pair order, and replaces non-overlapping occurrences left to right. Encoding always applies the available merge with the lowest learned rank. It is not greedy longest-token matching. Regex BPE performs the same algorithm independently inside regex chunks; the pattern must cover the complete input. This makes pretokenization part of the saved model.

Python and Rust must produce identical Basic merge tables and IDs for identical training data. Their saved JSON is interchangeable. UTF-8 text must round-trip, including combining characters, CRLF, tabs and NUL. Python rejects floating-point, boolean, negative and unknown token IDs. Rust's typed API uses usize; its Python binding also rejects boolean, float, negative and unknown IDs. Invalid UTF-8 token subsequences cannot be decoded as strict text; Python decode_bytes is available for inspection.

Training size is an upper bound. Fewer merges are learned when no pairs remain. Regex boundaries and training sample boundaries affect the learned vocabulary; a smaller token count is not proof of better language-model quality.

## SentencePiece adapter

The class name Llama2Tokenizer is retained for compatibility, but the class wraps the external SentencePiece library. Custom training does not recreate original Llama IDs or its complete prompt/BOS/EOS policy.

Thoth's custom training uses identity normalization, preserved spaces, no dummy prefix, full character coverage and byte fallback. A vocabulary needs at least 259 slots for bytes plus special tokens, and additional slots for observed characters. Insufficient sizes fail rather than silently disabling fallback.

Identity normalization alone does not guarantee reversibility: SentencePiece escapes spaces using U+2581, so a literal U+2581 can collide with its whitespace representation. encode first checks the model's ordinary result. If decoding changes the input, it encodes the entire input with explicit BYTE-type pieces, verifies exact recovery, and returns those IDs. If the loaded model cannot represent the input exactly, encode raises ValueError. The rare fallback increases token counts and differs from canonical SentencePiece encoding. Call processor.encode directly only when intentionally choosing the external model's normalization semantics; do not substitute Thoth's fallback IDs into an LLM expecting canonical tokenization.

Normal, lossless model output keeps its original IDs. Stored .model files remain standard SentencePiece artifacts. save writes the complete model; encode_batch uses the same checked contract per input. The raw processor remains available for model inspection.

References:
- [SentencePiece normalization and whitespace handling](https://github.com/google/sentencepiece/blob/master/doc/normalization.md)
- [SentencePiece model format and byte fallback settings](https://github.com/google/sentencepiece/blob/master/src/sentencepiece_model.proto)
- [SentencePiece paper](https://aclanthology.org/D18-2012/)

## Validation and performance

Regression tests cover literal whitespace markers, unseen emoji, NUL, combining characters, invalid IDs, model reload, and rejection of lossy external models. Property tests exercise arbitrary Unicode scalar values; isolated surrogate code points are outside the UTF-8 contract.

Run benchmarks/python_rust.py with a release-built native extension. The benchmark checks identical IDs and decoding before timing. It measures Python-to-native conversion costs, excludes model loading, warms each operation, and reports seven-trial medians and ranges. It uses repeated synthetic English, Python code and Unicode at several sizes; it is a microbenchmark, not a production corpus throughput claim. Training is measured separately at the same vocabulary target.

Current implementations repeatedly scan sequences while merging. Long inputs may scale poorly. Native methods currently retain the GIL, and batch processing is sequential. Neither concurrent speedup nor streaming tokenization is claimed. Tokenizing independent chunks can change IDs at chunk boundaries; do not present naive chunking as equivalent streaming.

The unfinished Rust Llama prototype is preserved under archive/rust, outside the active crate.
