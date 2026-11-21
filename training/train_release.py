"""Reproducible CPython-code/technical-English tokenizer training and evaluation."""

import hashlib, heapq, json, sys, tarfile, time, zipfile
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python/src"))
from regexTokenizer import RegexTokenizer
from thoth import load


def train_counts(chunks, size, progress=None):
    t = RegexTokenizer()
    words = list(chunks)
    weights = list(chunks.values())
    counts = Counter()
    locations = defaultdict(set)
    for i, word in enumerate(words):
        for pair, n in Counter(zip(word, word[1:])).items():
            counts[pair] += n * weights[i]
            locations[pair].add(i)
    heap = [(-n, pair) for pair, n in counts.items()]
    heapq.heapify(heap)
    for token in range(256, size):
        while heap:
            negative, pair = heapq.heappop(heap)
            if counts[pair] > 0 and -negative == counts[pair]:
                break
        else:
            break
        t.mints[pair] = token
        t.vocab[token] = t.vocab[pair[0]] + t.vocab[pair[1]]
        changed = set()
        for i in list(locations[pair]):
            old = Counter(zip(words[i], words[i][1:]))
            words[i] = tuple(t.merge(words[i], pair, token))
            new = Counter(zip(words[i], words[i][1:]))
            for p in old.keys() | new.keys():
                counts[p] += (new[p] - old[p]) * weights[i]
                changed.add(p)
                if p in new:
                    locations[p].add(i)
                else:
                    locations[p].discard(i)
        for p in changed:
            if counts[p] > 0:
                heapq.heappush(heap, (-counts[p], p))
        if len(heap) > max(100000, len(counts) * 4):
            heap = [(-n, p) for p, n in counts.items() if n > 0]
            heapq.heapify(heap)
        if progress and token % 1000 == 0:
            progress(token)
    return t


def main():
    root = Path(__file__).resolve().parents[1]
    archive = root / "training/cache/cpython-v3.13.0.tar.gz"
    expected = {
        "cpython-v3.13.0.tar.gz": "d2c5dbd73111a80ee654489eb0f52b663b38ef9ce9e9323312515d941d0dfaa2",
        "gutenberg.zip": "2d3c3ab548c653944310f37f536443ec85d0a0ad855fcae217a0c9efdce2d611",
    }
    for filename, digest in expected.items():
        if (
            hashlib.sha256(
                (root / "training/cache" / filename).read_bytes()
            ).hexdigest()
            != digest
        ):
            raise ValueError(f"Source archive checksum mismatch: {filename}")
    chunks = Counter()
    manifest = []
    held = []
    train_bytes = 0
    test_bytes = 0
    t = RegexTokenizer()
    seen = set()
    started = time.time()
    with tarfile.open(archive) as tar:
        for member in sorted(tar.getmembers(), key=lambda m: m.name):
            name = member.name
            if (
                not member.isfile()
                or member.size > 300000
                or not (
                    ("/Lib/" in name and name.endswith(".py"))
                    or ("/Doc/" in name and name.endswith(".rst"))
                )
            ):
                continue
            raw = tar.extractfile(member).read()
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                continue
            digest = hashlib.sha256(raw).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            split = (
                "test"
                if int(hashlib.sha256(name.encode()).hexdigest()[:8], 16) % 10 == 0
                else "train"
            )
            if split == "test":
                held.append((name, text))
                test_bytes += len(raw)
            else:
                if train_bytes + len(raw) > 25_000_000:
                    continue
                chunks.update(tuple(c) for c in t.chunks(text))
                train_bytes += len(raw)
            manifest.append(
                {"path": name, "sha256": digest, "bytes": len(raw), "split": split}
            )
    prose_bytes = 0
    with zipfile.ZipFile(root / "training/cache/gutenberg.zip") as archive_prose:
        for name in sorted(archive_prose.namelist()):
            if not name.endswith(".txt"):
                continue
            raw = archive_prose.read(name)
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = raw.decode("latin-1")
            split = (
                "test"
                if name.endswith(("austen-persuasion.txt", "chesterton-thursday.txt"))
                else "train"
            )
            if split == "test":
                held.append((name, text))
                test_bytes += len(raw)
            else:
                prose_bytes += len(raw)
                train_bytes += len(raw)
                chunks.update(
                    {
                        chunk: count * 3
                        for chunk, count in Counter(
                            tuple(c) for c in t.chunks(text)
                        ).items()
                    }
                )
            manifest.append(
                {
                    "path": name,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "bytes": len(raw),
                    "split": split,
                    "weight": 3 if split == "train" else 1,
                }
            )
    print(
        json.dumps(
            {
                "train_bytes": train_bytes,
                "test_bytes": test_bytes,
                "unique_chunks": len(chunks),
                "files": len(manifest),
            }
        ),
        flush=True,
    )
    t = train_counts(
        chunks,
        16384,
        lambda n: print(f"Trained {n} tokens ({time.time()-started:.1f}s)", flush=True),
    )
    path = root / "models/thoth-code-en-v1.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "kind": "regex",
                "pattern": t.pattern,
                "merges": list(t.mints),
            },
            separators=(",", ":"),
        )
    )
    saved = load(path)
    import tiktoken

    reference = tiktoken.get_encoding("cl100k_base")
    totals = {"bytes": 0, "thoth_tokens": 0, "cl100k_tokens": 0, "files": 0}
    groups = {}
    roundtrips = 0
    for name, text in held:
        ids = saved.encode(text)
        assert saved.decode(ids) == text
        assert ids == t.encode(text)
        group = groups.setdefault(
            "prose" if name.startswith("gutenberg/") else "code_and_docs",
            {"bytes": 0, "thoth_tokens": 0, "cl100k_tokens": 0},
        )
        group["bytes"] += len(text.encode())
        group["thoth_tokens"] += len(ids)
        group["cl100k_tokens"] += len(reference.encode_ordinary(text))
        roundtrips += 1
        totals["bytes"] += len(text.encode())
        totals["thoth_tokens"] += len(ids)
        totals["cl100k_tokens"] += len(reference.encode_ordinary(text))
        totals["files"] += 1
    report = {
        "name": "Thoth English + Code v1",
        "vocabulary_size": len(t.vocab),
        "training_bytes": train_bytes,
        "prose_training_bytes": prose_bytes,
        "prose_weight": 3,
        "prose_archive_sha256": hashlib.sha256(
            (root / "training/cache/gutenberg.zip").read_bytes()
        ).hexdigest(),
        "held_out": totals,
        "held_out_groups": groups,
        "roundtrip_files": roundtrips,
        "seconds": time.time() - started,
        "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "source": [
            "https://github.com/python/cpython/tree/v3.13.0",
            "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/gutenberg.zip",
        ],
        "split": "CPython: SHA256(path) modulo 10 == 0 held out; duplicate file contents excluded. Gutenberg: Persuasion and The Man Who Was Thursday held out whole.",
        "files": manifest,
    }
    (root / "models/thoth-code-en-v1.report.json").write_text(
        json.dumps(report, indent=2)
    )
    print(
        json.dumps({k: v for k, v in report.items() if k != "files"}, indent=2),
        flush=True,
    )
    with tarfile.open(archive) as tar:
        (root / "models/CPython-LICENSE.txt").write_bytes(
            tar.extractfile("cpython-3.13.0/LICENSE").read()
        )


if __name__ == "__main__":
    main()
