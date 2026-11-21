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

`Llama2Tokenizer(model_file='tokenizer.model')` requires the original model for exact compatibility. Its optional training method creates a new SentencePiece model, not a pretrained Llama vocabulary. The adapter checks exact text recovery, uses explicit byte pieces when needed and available, and raises an error if a loaded model cannot preserve the input. This checked mode can differ from canonical Llama/SentencePiece IDs on exceptional text; see the tokenizer contracts.

## Playground

```sh
cd playground
npm ci
npm run dev
```

Open the printed localhost URL and start the local backend as described below. The default is Thoth's trained Regex BPE. Four local implementations load saved vocabularies; one JavaScript reference, cl100k_base, runs in a browser worker. Inspect colored tokens and raw bytes, display whitespace, or compare counts. The readable byte-BPE view preserves whole graphemes, including emoji and combining marks. Select text or a token to inspect their relationship and exact UTF-8 offsets. Counts exclude chat framing. Input is limited to 20,000 UTF-8 bytes.

Local temporary training uses the editor text and never overwrites saved release models. SentencePiece also accepts model uploads. Its decoded view follows the loaded model's normalization and does not claim byte alignment.

`npm run build` creates static assets in `playground/dist`. The reference table loads on demand in a module worker. Local choices require the loopback backend; no API key is needed. Inspired by [Tiktokenizer](https://tiktokenizer.vercel.app/).

## Model format

```json
{"version":1,"kind":"basic","merges":[[97,98],[256,256]]}
```

Base IDs 0–255 represent bytes. Each merge's array position assigns its ID starting at 256; references must identify earlier tokens. Python regex models also carry `kind: "regex"` and `pattern`. Rust supports this basic persistence format through its library and optional Python bindings.

## Validation and benchmarks

```sh
pytest -q
cargo test --manifest-path rust/Cargo.toml
cd playground
npx playwright install chromium
npm run test:e2e
```

The tests cover deterministic merges, Unicode round trips, invalid IDs/models, CLI subprocesses, reference encoding parity, SentencePiece model loading, and desktop/mobile browser interactions. `thoth compare` reports encoding time and token counts; custom BPE entries train on the input corpus, while named encodings are pretrained, so compression results are not a controlled model-quality comparison. Timing excludes training and encoding initialization.

Rust demo: `cargo run --manifest-path rust/Cargo.toml --bin thoth -- 'Hello 🦉'`. The Rust executable exposes the working basic BPE implementation; the completed SentencePiece/Llama runtime is available as a Rust library and native Python class. The train_bpe executable uses incremental weighted pair counts for release training.

## Optional native Python backend

With Rust installed, build the extension in an active virtual environment:

```sh
pip install ./rust
```

`from thoth_native import NativeTokenizer` exposes `train`, `encode`, `decode`, `encode_batch`, `to_json`, and `from_json`. Basic models interchange with the CLI and local playground backend; regex models are rejected by the native backend. Python regex training weights repeated chunks to avoid scanning duplicate chunks at every merge. JSONL CLI encoding processes one line at a time. Comparison reports include peak Python-tracked allocations; these exclude native-library allocations and are not total process memory. Native bindings currently retain the GIL; batch APIs are sequential.

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

The dropdown exposes Python BasicTokenizer, RegexTokenizer, the native Rust BasicTokenizer, and the SentencePiece adapter through a loopback-only service on port 8765, proxied by Vite. Each loads a saved 16K vocabulary. JavaScript cl100k_base is the sole reference choice. The Rust SentencePiece runtime is available through the library; it has not yet been added to the dropdown.

Temporary models are held in memory (up to 16) and expire when the server restarts or evicts them; release artifacts persist on disk. Browser cancellation ignores an in-flight server result; it does not interrupt native computation already running. Standalone static hosting supports the JavaScript reference only; local choices show a backend setup error if the service is unavailable.

Run `THOTH_LOCAL_TESTS=1 npm run test:e2e` with the backend running to exercise all local dropdown options end to end.

## Saved English + code tokenizer

The playground includes **Regex BPE · Python · trained 16K**, backed by the committed model in `models/thoth-code-en-v1.json`. See [the model card](models/README.md) for the 36 MB training corpus, held-out results, limitations, source licenses and reproduction commands. No retraining is needed to use this preset.

## Trained tokenizer family

The UI now has four local choices and one browser reference: Python Basic BPE, Rust Basic BPE, Python Regex BPE, custom SentencePiece BPE, and JavaScript `cl100k_base`. Each local choice loads a saved 16K vocabulary by default. Basic Python and Rust deliberately share one vocabulary; different implementation languages should not change token IDs. Temporary demo training creates a separate session model and never modifies release artifacts.

`python training/train_family.py` trains Basic and SentencePiece from the exact file manifest and split of the Regex release, then evaluates all implementations on the same held-out files. Basic uses independent complete lines as training samples; SentencePiece uses lines split at 2048 characters. See `models/family-evaluation.json` for measured results. The SentencePiece preset is a new custom vocabulary, not the original Llama tokenizer. JavaScript BPE training and the extra reference encodings have been removed from the dropdown.

## Correctness contracts and benchmarks

See [tokenizer contracts](docs/tokenizer-contracts.md) for merge ordering, Unicode behavior, SentencePiece's checked byte fallback, external-model compatibility, and remaining performance limits. The Rust Llama prototype has now been completed; see [its API and design](docs/rust-sentencepiece.md).

Run `python benchmarks/python_rust.py` after installing a release-built native extension (`pip install ./rust`). Results are written to `benchmarks/python-rust-results.json`; the comparison uses identical saved vocabularies and checks IDs before measuring.

The native SentencePiece class is thoth_native.NativeLlama2Tokenizer. It supports standard model loading, checked/canonical encoding, optional BOS/EOS, decoding, batches, saving, and native custom BPE training. [Full examples and limits](docs/rust-sentencepiece.md).

## TODO: tokenizer roadmap

Proposals for future work, not promises of compatibility or performance. Prioritize the tokenizer library before expanding the playground. Check an item off only when its behavior is documented, regression tests pass, and any performance claims have reproducible evidence.

### Correctness and compatibility

- [ ] **Validate original Llama model compatibility.** Add an opt-in fixture for a user-supplied original model and compare canonical IDs, normalization, BOS/EOS, and decoding against the official reference; keep custom vocabularies clearly distinct.
- [ ] **Expand adversarial testing.** Fuzz model imports and Unicode inputs, including malformed protobufs, pathological regex patterns, byte-token boundaries, and user-defined/unused SentencePiece pieces; require bounded failures instead of crashes or hangs.
- [ ] **Unify tokenizer interfaces.** Define consistent validation, error types, batch methods, model metadata, and explicit canonical/lossless modes across Python and Rust without silently changing existing token IDs.
- [ ] **Track model provenance.** Version training settings, corpus licenses and hashes, preprocessing, vocabulary checksums, and evaluation splits alongside every release artifact.

### Training, evaluation, and performance

- [ ] **Scale native SentencePiece-format training.** Replace repeated full-corpus pair counting with incremental weighted updates, measure memory use, and add progress/cancellation and resumable checkpoints before increasing the current training limits.
- [ ] **Broaden the English/code corpus.** Add licensed modern prose and multiple programming languages, deduplicate across train/test splits, and evaluate vocabulary sizes on held-out domains before choosing a new release.
- [ ] **Benchmark realistic workloads.** Measure throughput, p50/p95 latency, peak memory, model-load cost, long-input scaling, and batches on real held-out files; compare identical model IDs and distinguish pure Python from Python wrappers around C++.
- [ ] **Optimize Basic BPE and batching.** Evaluate ranked merge queues, releasing the Rust binding's GIL, and parallel batches; retain deterministic output and prove gains against the existing baseline.
- [ ] **Add multilingual quality reports.** Report tokens per UTF-8 byte, fallback/unknown-token rates, and exact-recovery rates by language and code domain, including normalization-sensitive text.

### Tokenizer extensions

- [ ] **Implement Rust Regex BPE.** Load the same pattern and merges as Python and verify matching split boundaries and token IDs, including Unicode properties and regex-dialect differences.
- [ ] **Expose token offsets.** Return byte spans and special-token metadata; explicitly represent normalized or synthetic output rather than inventing exact input offsets.
- [ ] **Investigate exact streaming.** Define buffering and boundary rules that preserve whole-input tokenization; test arbitrary chunk boundaries and document models for which bounded buffering cannot guarantee identical IDs.
- [ ] **Add optional BPE dropout.** Support seeded stochastic segmentation for training while keeping ordinary inference deterministic and model-compatible.
- [ ] **Explore native Unigram or WordPiece training.** Write a design and reference-parity plan first; distinguish new vocabulary training from the existing Unigram model-loading support.

For complex changes, add a design note under [docs](docs/) covering algorithm choices, model-format impact, compatibility, failure modes, and acceptance tests. Start from the [tokenizer contracts](docs/tokenizer-contracts.md), [Rust SentencePiece design](docs/rust-sentencepiece.md), and [benchmark baseline](benchmarks/README.md).
