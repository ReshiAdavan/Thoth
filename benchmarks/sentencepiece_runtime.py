"""Compare checked SentencePiece APIs: Python/C++ upstream versus pure Rust."""

import hashlib
import json
import platform
from pathlib import Path
from python_rust import measure, REPEATS
from llama2Tokenizer import Llama2Tokenizer
from thoth_native import NativeLlama2Tokenizer

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "models/thoth-sentencepiece-en-code-v1.model"
python = Llama2Tokenizer(path)
rust = NativeLlama2Tokenizer(str(path))
rows = []
for name, seed in {
    "english": "The curious reader explores language and meaning.\n",
    "code": "def total(items):\n    return sum(x.price for x in items)\n",
    "unicode": "Hello 🦉 中文 e\u0301 café\r\n",
}.items():
    for target in [256, 4096, 16384]:
        text = (
            (seed * (target // len(seed.encode()) + 1))
            .encode()[:target]
            .decode("utf-8", errors="ignore")
        )
        ids = python.encode(text)
        assert rust.encode(text) == ids
        assert python.decode(ids) == rust.decode(ids) == text
        for operation, arg in [("encode", text), ("decode", ids)]:
            p = measure(lambda: getattr(python, operation)(arg))
            r = measure(lambda: getattr(rust, operation)(arg))
            rows.append(
                {
                    "corpus": name,
                    "bytes": len(text.encode()),
                    "operation": operation,
                    "python_cpp": p,
                    "rust": r,
                    "rust_speedup": p["median_ms"] / r["median_ms"],
                }
            )
report = {
    "platform": platform.platform(),
    "repeats": REPEATS,
    "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    "method": "Checked lossless APIs; Python adapter runs Google's C++ SentencePiece, Rust adapter runs Thoth heap-based BPE with sentencepiece-rs normalization/decoding. Same saved model and IDs, release build, warm-up and seven trials. Synthetic repeated inputs. Training is not compared because the two trainers implement different procedures.",
    "results": rows,
}
(ROOT / "benchmarks/sentencepiece-results.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
for row in rows:
    print(
        row["corpus"],
        row["bytes"],
        row["operation"],
        round(row["rust_speedup"], 2),
        "x",
    )
