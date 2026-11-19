import os
import subprocess
import sys
import json
import pytest
from basicTokenizer import BasicTokenizer
from thoth import save, load


def run(*args, text=""):
    env = dict(os.environ, PYTHONPATH="python/src")
    return subprocess.run(
        [sys.executable, "-m", "thoth", *args],
        input=text,
        text=True,
        capture_output=True,
        env=env,
    )


def test_cli(tmp_path):
    model = str(tmp_path / "model.json")
    assert (
        run(
            "train", "--output", model, "--vocab-size", "280", text="hello hello 🦉"
        ).returncode
        == 0
    )
    encoded = run("encode", "--model", model, text="hello 中文 🦉\n")
    decoded = run("decode", "--model", model, text=encoded.stdout)
    assert decoded.stdout == "hello 中文 🦉\n"
    assert run("decode", "--model", model, text="[999999]").returncode == 2
    assert (
        run("encode", "--model", model, "--jsonl", text="hello\nworld").stdout.count(
            "\n"
        )
        == 2
    )


def test_invalid_model(tmp_path):
    path = tmp_path / "model.json"
    path.write_text(json.dumps({"version": 1, "kind": "basic", "merges": [[256, 0]]}))
    with pytest.raises(ValueError):
        load(path)


def test_save_load(tmp_path):
    t = BasicTokenizer()
    t.train("abababab", 270)
    p = tmp_path / "model.json"
    save(t, p)
    assert load(p).encode("abababab") == t.encode("abababab")
