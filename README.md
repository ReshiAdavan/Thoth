# Thoth

A tokenizer library, command-line tool and browser playground for exploring byte-pair encoding. Python and Rust implement deterministic byte BPE; named pretrained encodings use tiktoken. A SentencePiece adapter loads actual model files for Llama-compatible IDs.

## Install and use

Python 3.10+:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[test,sentencepiece]'
printf 'hello hello world' | thoth train --tokenizer basic --vocab-size 300 --output model.json
printf 'hello world' | thoth encode --model model.json
printf '[104,101,108,108,111]' | thoth decode --tokenizer basic
thoth inspect --tokenizer cl100k_base --input data/taylorSwift.txt
thoth compare --input data/taylorSwift.txt
```

Commands accept `--input` and `--output`; omitted input reads stdin. Encode emits JSON IDs. Decode emits exact text without adding a newline. `encode --jsonl` handles each line independently. Named encodings: `cl100k_base`, `o200k_base`, `p50k_base`, `r50k_base`. First Python use may download encoding data. Text resembling special tokens is treated as ordinary text unless explicitly allowed through the Python API.

`BasicTokenizer` and `RegexTokenizer` provide `train`, `encode`, `decode`, `decode_bytes`, and `encode_batch`; legacy `encoder`/`decoder` aliases remain. Empty encoding/decoding is supported; training rejects empty text. Invalid IDs and invalid UTF-8 decoding raise errors. Training stops when no pairs remain, so the requested vocabulary is an upper bound. Retraining resets the model. Equal-frequency pairs use lexicographic ordering.

`Llama2Tokenizer(model_file='tokenizer.model')` requires the original model for exact compatibility. Its optional training method creates a new SentencePiece model, not a pretrained Llama vocabulary. SentencePiece normalization and unknown-token behavior follow the loaded model; arbitrary unseen text is not guaranteed lossless.

## Playground

```sh
cd playground
npm ci
npm run dev
```

Open the printed localhost URL. Tokenization runs in the browser. Choose an encoding, inspect colored tokens and raw bytes, display whitespace, compare counts, or train a byte BPE vocabulary. Export/import basic models using the same versioned JSON format as the Python CLI. The readable view preserves whole graphemes (including emoji and combining marks). Select text or a token to highlight their relationship; token inspection reports exact UTF-8 byte offsets. Tokens containing partial UTF-8 characters display hex bytes instead of replacement glyphs. Counts exclude chat framing and other model-specific overhead. The interactive input is limited to 20,000 UTF-8 bytes; custom vocabulary limits are 256–1024.

`npm run build` creates static assets in `playground/dist`. Encoding tables load on demand in a module worker. Tokenization, training and imports run off the UI thread; newer requests or Cancel terminate in-flight work. Idle workers retain loaded encodings. No application backend or API key is needed. Inspired by [Tiktokenizer](https://tiktokenizer.vercel.app/).

## Model format

```json
{"version":1,"kind":"basic","merges":[[97,98],[256,256]]}
```

Base IDs 0–255 represent bytes. Each merge's array position assigns its ID starting at 256; references must identify earlier tokens. Python regex models also carry `kind: "regex"` and `pattern`. The browser currently imports basic models only. Rust supports this basic persistence format through its library and optional Python bindings.

## Validation and benchmarks

```sh
pytest -q
cargo test --manifest-path rust/Cargo.toml
cd playground
npx playwright install chromium
npm run test:e2e
```

The tests cover deterministic merges, Unicode round trips, invalid IDs/models, CLI subprocesses, reference encoding parity, SentencePiece model loading, and desktop/mobile browser interactions. `thoth compare` reports encoding time and token counts; custom BPE entries train on the input corpus, while named encodings are pretrained, so compression results are not a controlled model-quality comparison. Timing excludes training and encoding initialization.

Rust demo: `cargo run --manifest-path rust/Cargo.toml -- 'Hello 🦉'`. The Rust executable exposes the working basic BPE implementation; the historical unfinished Llama implementation is not compiled. Incremental pair-count updates for large-corpus training remain a future optimization.

## Optional native Python backend

With Rust installed, build the extension in an active virtual environment:

```sh
pip install ./rust
```

`from thoth_native import NativeTokenizer` exposes `train`, `encode`, `decode`, `encode_batch`, `to_json`, and `from_json`. Basic models interchange with the CLI and browser; regex models are rejected by the native backend. Python regex training weights repeated chunks to avoid scanning duplicate chunks at every merge. JSONL CLI encoding processes one line at a time. Comparison reports include peak Python-tracked allocations; these exclude native-library allocations and are not total process memory. Native bindings currently retain the GIL; batch APIs are sequential.

## Model import bounds

Python, Rust and the browser reject files over 1,000,000 bytes, more than 100,000 merges, token expansions over 65,536 bytes, or a total expanded vocabulary over 16,777,216 bytes. Expansion is checked before allocation. A shared fixture corpus exercises malformed references, duplicate merges and exponential expansion across all three implementations.

Use `THOTH_PREVIEW=1 npm run test:e2e` in `playground` to run browser checks against the production build rather than the development server.

## Show the actual local implementations in the UI

Run these in separate terminals from the checkout:

```sh
source .venv/bin/activate
pip install -e '.[test,sentencepiece]'
# Optional Rust implementation requires a Rust toolchain:
pip install ./rust
python -m thoth_demo
```

```sh
cd playground
npm run dev
```

The dropdown groups browser reference encodings separately from Thoth implementations. Python BasicTokenizer, RegexTokenizer, GPT4Tokenizer, the native Rust BasicTokenizer, and the SentencePiece adapter execute through a loopback-only service on port 8765, proxied by Vite. Train Basic/Regex/Rust on the editor text; their initial vocabulary is bytes only. GPT4Tokenizer uses pretrained cl100k_base. SentencePiece supports demo training or original `.model` upload (1 MB maximum); custom training is not pretrained Llama compatibility. The unfinished historical Rust Llama implementation is not exposed.

Local models are held in memory (up to 16) and expire when the server restarts or evicts them. Browser cancellation ignores an in-flight server result; it does not interrupt native computation already running. The decoded SentencePiece view intentionally does not claim exact input-byte alignment because normalization and unknown tokens depend on the loaded model. Standalone static hosting supports browser choices only; local choices show a backend setup error if the service is unavailable.

Run `THOTH_LOCAL_TESTS=1 npm run test:e2e` with the backend running to exercise all local dropdown options end to end.

## Saved English + code tokenizer

The playground includes **Thoth English + Code v1 · trained 16K**, backed by the committed model in `models/thoth-code-en-v1.json`. See [the model card](models/README.md) for the 36 MB training corpus, held-out results, limitations, source licenses and reproduction commands. No retraining is needed to use this preset.
