"""Same-model Python versus release-mode Rust through the public Python API."""

import hashlib
import json
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python/src"))
from thoth import load
from basicTokenizer import BasicTokenizer
from thoth_native import NativeTokenizer

REPEATS = 7


def measure(call):
    call()  # warm-up, outside measurements
    samples = []
    for _ in range(REPEATS):
        start = time.perf_counter_ns()
        call()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def main():
    path = ROOT / "models/thoth-basic-en-code-v1.json"
    python = load(path)
    rust = NativeTokenizer.from_json(path.read_text())
    results = []
    seeds = {
        "english": "The curious reader explores language, meaning, and the way words fit together.\n",
        "code": "def total(items):\n    return sum(item.price for item in items if item.active)\n",
        "unicode": "Hello 🦉 café e\u0301 中文 नमस्ते مرحبا\t\r\n",
    }
    for name, seed in seeds.items():
        for target in [256, 4096, 16384]:
            text = seed * (target // len(seed.encode()) + 1)
            text = text.encode()[:target].decode("utf-8", errors="ignore")
            ids = python.encode(text)
            assert rust.encode(text) == ids
            assert python.decode(ids) == rust.decode(ids) == text
            for operation in ["encode", "decode"]:
                arg = text if operation == "encode" else ids
                p = measure(lambda: getattr(python, operation)(arg))
                r = measure(lambda: getattr(rust, operation)(arg))
                results.append(
                    {
                        "corpus": name,
                        "bytes": len(text.encode()),
                        "tokens": len(ids),
                        "operation": operation,
                        "python": p,
                        "rust": r,
                        "rust_speedup": p["median_ms"] / r["median_ms"],
                    }
                )
    corpus = (seeds["english"] + seeds["code"]) * 12
    p = BasicTokenizer()
    n = NativeTokenizer()
    p.train(corpus, 512)
    n.train(corpus, 512)
    assert list(p.mints) == [tuple(pair) for pair in json.loads(n.to_json())["merges"]]
    pt = measure(lambda: BasicTokenizer().train(corpus, 512))
    rt = measure(lambda: NativeTokenizer().train(corpus, 512))
    report = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "repeats": REPEATS,
        "method": "Warm-up + 7 sequential wall-clock trials; medians. Native extension installed in release mode. Python-to-Rust conversion included. Model loading excluded. Synthetic repeated text; no general throughput claim.",
        "results": results,
        "training": {
            "bytes": len(corpus.encode()),
            "target_vocab": 512,
            "python": pt,
            "rust": rt,
            "rust_speedup": pt["median_ms"] / rt["median_ms"],
        },
    }
    (ROOT / "benchmarks/python-rust-results.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    for row in results:
        print(
            row["corpus"],
            row["bytes"],
            row["operation"],
            round(row["rust_speedup"], 2),
            "x",
            flush=True,
        )
    print("training", report["training"], flush=True)


if __name__ == "__main__":
    main()
