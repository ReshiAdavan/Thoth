import importlib.util
from pathlib import Path
from collections import Counter
from regexTokenizer import RegexTokenizer
from thoth import load

spec = importlib.util.spec_from_file_location(
    "train_release", Path("training/train_release.py")
)
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)


def test_incremental_matches_reference():
    for text in [
        "abababab hello hello",
        "🦉 café 你好 123456\n",
        "aaaaaaa abc abc def def",
    ]:
        for size in [256, 260, 280]:
            reference = RegexTokenizer()
            reference.train(text, size)
            trained = trainer.train_counts(
                Counter(tuple(c) for c in reference.chunks(text)), size
            )
            assert trained.mints == reference.mints
            assert trained.encode(text) == reference.encode(text)


def test_release_model():
    model = load("models/thoth-code-en-v1.json")
    assert len(model.vocab) == 16384
    for text in ["Hello world!", "def f(x):\n    return x + 1", "🦉 e\u0301 中文"]:
        assert model.decode(model.encode(text)) == text
