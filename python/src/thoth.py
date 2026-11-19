"""Thoth CLI and versioned, portable BPE model format."""

import argparse
import json
import sys
import time
import tracemalloc
from pathlib import Path
from basicTokenizer import BasicTokenizer
from regexTokenizer import RegexTokenizer
from gpt4Tokenizer import GPT4Tokenizer


def save(tokenizer, path):
    data = {
        "version": 1,
        "kind": "regex" if isinstance(tokenizer, RegexTokenizer) else "basic",
        "merges": [list(pair) for pair in tokenizer.mints],
    }
    if isinstance(tokenizer, RegexTokenizer):
        data["pattern"] = tokenizer.pattern
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        not isinstance(data, dict)
        or data.get("version") != 1
        or data.get("kind") not in ("basic", "regex")
    ):
        raise ValueError("Unsupported model format")
    if not isinstance(data.get("merges"), list):
        raise ValueError("Model merges must be an array")
    if data["kind"] == "regex" and not isinstance(data.get("pattern"), str):
        raise ValueError("Regex model needs a pattern")
    t = RegexTokenizer(data["pattern"]) if data["kind"] == "regex" else BasicTokenizer()
    for token, pair in enumerate(data["merges"], 256):
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or any(type(i) is not int or i not in t.vocab for i in pair)
        ):
            raise ValueError("Invalid merge: references must point to earlier tokens")
        pair = tuple(pair)
        if pair in t.mints:
            raise ValueError("Duplicate merge")
        t.mints[pair] = token
        t.vocab[token] = t.vocab[pair[0]] + t.vocab[pair[1]]
    return t


def select(name="cl100k_base", model=None):
    if model:
        return load(model)
    if name == "basic":
        return BasicTokenizer()
    if name == "regex":
        return RegexTokenizer()
    return GPT4Tokenizer(name)


def main():
    parser = argparse.ArgumentParser(
        prog="thoth", description="Train, inspect and compare tokenizers"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("train", "encode", "decode", "inspect", "compare"):
        p = sub.add_parser(command)
        p.add_argument("--input", help="UTF-8 input file (default: stdin)")
        p.add_argument("--output", help="Output file (default: stdout)")
        p.add_argument(
            "--tokenizer",
            default="cl100k_base",
            choices=[
                "basic",
                "regex",
                "cl100k_base",
                "o200k_base",
                "p50k_base",
                "r50k_base",
            ],
        )
        p.add_argument("--model", help="Saved Thoth model")
        if command == "train":
            p.set_defaults(tokenizer="regex")
            p.add_argument("--vocab-size", type=int, default=1000)
        if command == "encode":
            p.add_argument(
                "--jsonl", action="store_true", help="Encode each line independently"
            )
    args = parser.parse_args()
    try:
        if (
            args.input
            and args.output
            and Path(args.input).resolve() == Path(args.output).resolve()
        ):
            raise ValueError("Input and output must be different files")
        if args.command == "encode" and args.jsonl:
            tokenizer = select(args.tokenizer, args.model)
            source = open(args.input, encoding="utf-8") if args.input else sys.stdin
            target = (
                open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
            )
            try:
                for line in source:
                    target.write(json.dumps(tokenizer.encode(line.rstrip("\r\n"))))
                    target.write("\n")
            finally:
                if args.input:
                    source.close()
                if args.output:
                    target.close()
            return
        text = (
            Path(args.input).read_text(encoding="utf-8")
            if args.input
            else sys.stdin.read()
        )
        tokenizer = select(args.tokenizer, args.model)
        if args.command == "train":
            if not args.output:
                raise ValueError("Training requires --output MODEL.json")
            if not isinstance(tokenizer, BasicTokenizer):
                raise ValueError("Train supports basic or regex")
            tokenizer.train(text, args.vocab_size)
            save(tokenizer, args.output)
            return
        if args.command == "decode":
            ids = json.loads(text)
            if not isinstance(ids, list) or any(
                type(i) is not int or i < 0 for i in ids
            ):
                raise ValueError("Expected a JSON array of nonnegative token IDs")
            result = tokenizer.decode(ids)
        elif args.command == "compare":
            rows = []
            for name in ["basic", "regex", "cl100k_base", "o200k_base"]:
                t = select(name)
                if isinstance(t, BasicTokenizer) and text:
                    t.train(text, 512)
                tracemalloc.start()
                start = time.perf_counter()
                ids = t.encode(text)
                elapsed = time.perf_counter() - start
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                rows.append(
                    {
                        "tokenizer": name,
                        "tokens": len(ids),
                        "utf8_bytes": len(text.encode()),
                        "encode_seconds": elapsed,
                        "python_peak_bytes": peak,
                        "training": (
                            "input corpus"
                            if isinstance(t, BasicTokenizer)
                            else "pretrained"
                        ),
                    }
                )
            result = json.dumps(rows, indent=2)
        elif args.command == "inspect":
            ids = tokenizer.encode(text)
            result = json.dumps(
                [{"id": i, "bytes": list(tokenizer.decode_bytes([i]))} for i in ids],
                indent=2,
            )
        else:
            result = json.dumps(tokenizer.encode(text))
        if args.output:
            Path(args.output).write_text(result, encoding="utf-8")
        else:
            sys.stdout.write(result)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(2, f"thoth: {error}\n")


if __name__ == "__main__":
    main()
