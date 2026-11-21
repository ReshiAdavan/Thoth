"""Train Basic BPE and SentencePiece on the same release manifest as Regex BPE."""

import hashlib, json, sys, tarfile, zipfile, time, subprocess
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python/src"))
from thoth import load
import sentencepiece as spm

ROOT = Path(__file__).resolve().parents[1]


def documents():
    report = json.loads((ROOT / "models/thoth-code-en-v1.report.json").read_text())
    tar_path = ROOT / "training/cache/cpython-v3.13.0.tar.gz"
    zip_path = ROOT / "training/cache/gutenberg.zip"
    assert hashlib.sha256(tar_path.read_bytes()).hexdigest() == report["archive_sha256"]
    assert (
        hashlib.sha256(zip_path.read_bytes()).hexdigest()
        == report["prose_archive_sha256"]
    )
    with tarfile.open(tar_path) as tar, zipfile.ZipFile(zip_path) as archive:
        for entry in report["files"]:
            raw = (
                archive.read(entry["path"])
                if entry["path"].startswith("gutenberg/")
                else tar.extractfile(entry["path"]).read()
            )
            assert hashlib.sha256(raw).hexdigest() == entry["sha256"]
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = raw.decode("latin-1")
            yield entry, text


def sentences():
    for entry, text in documents():
        if entry["split"] != "train":
            continue
        for _ in range(entry.get("weight", 1)):
            for line in text.splitlines():
                for start in range(0, len(line), 2048):
                    yield line[start : start + 2048]


def main():
    started = time.time()
    chunks = Counter()
    for entry, text in documents():
        if entry["split"] == "train":
            for line, count in Counter(text.splitlines(keepends=True)).items():
                chunks[tuple(line.encode())] += count * entry.get("weight", 1)
    print(f"Basic training: {len(chunks)} distinct line samples", flush=True)
    samples = ROOT / "training/cache/basic-samples.jsonl"
    with samples.open("w") as out:
        for sample, weight in chunks.items():
            out.write(json.dumps([bytes(sample).decode("utf-8"), weight]) + "\n")
    path = ROOT / "models/thoth-basic-en-code-v1.json"
    subprocess.run(
        [
            "cargo",
            "run",
            "--release",
            "--manifest-path",
            str(ROOT / "rust/Cargo.toml"),
            "--bin",
            "train_bpe",
            "--",
            str(samples),
            str(path),
        ],
        check=True,
    )
    del chunks
    print("Training SentencePiece BPE", flush=True)
    spm.SentencePieceTrainer.train(
        sentence_iterator=sentences(),
        model_prefix=str(ROOT / "models/thoth-sentencepiece-en-code-v1"),
        vocab_size=16384,
        model_type="bpe",
        hard_vocab_limit=False,
        normalization_rule_name="identity",
        remove_extra_whitespaces=False,
        add_dummy_prefix=False,
        byte_fallback=True,
        character_coverage=1.0,
        num_threads=2,
        shuffle_input_sentence=False,
        max_sentence_length=16384,
    )
    print("Evaluating saved models", flush=True)
    basic = load(path)
    regex = load(ROOT / "models/thoth-code-en-v1.json")
    from thoth_native import NativeTokenizer

    native = NativeTokenizer.from_json(path.read_text())
    sp = spm.SentencePieceProcessor(
        model_file=str(ROOT / "models/thoth-sentencepiece-en-code-v1.model")
    )
    import tiktoken

    ref = tiktoken.get_encoding("cl100k_base")
    results = {}
    files = 0
    for entry, text in documents():
        if entry["split"] != "test":
            continue
        group = "prose" if entry["path"].startswith("gutenberg/") else "code_and_docs"
        row = results.setdefault(
            group,
            {
                "bytes": 0,
                "basic": 0,
                "rust": 0,
                "regex": 0,
                "sentencepiece": 0,
                "cl100k_base": 0,
                "roundtrips": {"basic": 0, "rust": 0, "regex": 0, "sentencepiece": 0},
                "files": 0,
            },
        )
        # Basic was trained on independent lines; no cross-line merges exist.
        ids = [i for line in text.splitlines(keepends=True) for i in basic.encode(line)]
        rust = [
            i for line in text.splitlines(keepends=True) for i in native.encode(line)
        ]
        assert ids == rust
        regex_ids = regex.encode(text)
        sp_ids = sp.encode(text, out_type=int)
        row["bytes"] += len(text.encode())
        row["files"] += 1
        for name, values in [
            ("basic", ids),
            ("rust", rust),
            ("regex", regex_ids),
            ("sentencepiece", sp_ids),
            ("cl100k_base", ref.encode_ordinary(text)),
        ]:
            row[name] += len(values)
        for name, decoded in [
            ("basic", basic.decode(ids)),
            ("rust", native.decode(rust)),
            ("regex", regex.decode(regex_ids)),
            ("sentencepiece", sp.decode(sp_ids)),
        ]:
            row["roundtrips"][name] += decoded == text
        files += 1
        if files % 25 == 0:
            print(f"Evaluated {files} files", flush=True)
    report = {
        "vocabulary_target": 16384,
        "basic_training_boundaries": "independent complete lines, not regex splitting",
        "sentencepiece_training_boundaries": "lines split at 2048 Unicode characters; blank lines omitted by iterator",
        "source_manifest": "thoth-code-en-v1.report.json",
        "results": results,
        "seconds": time.time() - started,
        "artifacts": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [path, ROOT / "models/thoth-sentencepiece-en-code-v1.model"]
        },
    }
    (ROOT / "models/family-evaluation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
