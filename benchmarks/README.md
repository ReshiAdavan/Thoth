# Python versus Rust benchmark

Measured locally on macOS arm64 with Python 3.13 and a release-built Rust extension. Both use the same saved Basic BPE vocabulary; every input has matching IDs and exact round-trip verification. Results include native-call conversion and validation, exclude model loading, and use seven timed trials after warm-up.

| Input | Bytes | Python encode (ms) | Rust encode (ms) | Rust speedup |
|---|---:|---:|---:|---:|
| english | 256 | 1.035 | 0.169 | 6.14× |
| english | 4096 | 18.968 | 2.552 | 7.43× |
| english | 16384 | 77.151 | 10.518 | 7.33× |
| code | 256 | 0.888 | 0.146 | 6.08× |
| code | 4096 | 16.132 | 2.240 | 7.20× |
| code | 16384 | 67.473 | 8.941 | 7.55× |
| unicode | 256 | 0.173 | 0.029 | 5.95× |
| unicode | 4096 | 2.968 | 0.469 | 6.33× |
| unicode | 16384 | 15.671 | 2.388 | 6.56× |

Decoding was 2.52–4.77× faster in Rust. Training the same 1,884-byte corpus to a 512-token vocabulary took 16.675 ms in Python and 0.957 ms in Rust (17.43×). This small training test uses the public trainers, not the large-corpus incremental release trainer.

These repeated synthetic English/code/Unicode inputs are microbenchmarks. They do not establish a production SLA, representative large-corpus throughput, or multithreaded speedup. Both implementations still repeatedly scan sequences and retain limits described in [the contracts](../docs/tokenizer-contracts.md).

Reproduce with a release native installation and python benchmarks/python_rust.py. Full samples are summarized as median/min/max timings in [the machine-readable results](python-rust-results.json).
