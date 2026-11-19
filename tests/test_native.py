import pytest

native = pytest.importorskip("thoth_native")
from basicTokenizer import BasicTokenizer
from thoth import save, load


def test_native_interchange(tmp_path):
    p = tmp_path / "model.json"
    t = BasicTokenizer()
    t.train("abababab hello 🦉", 280)
    save(t, p)
    n = native.NativeTokenizer.from_json(p.read_text())
    texts = ["abababab", "hello 🦉", "中文", ""]
    assert n.encode_batch(texts) == t.encode_batch(texts)
    n.train("abababab hello 🦉", 280)
    assert n.encode_batch(texts) == t.encode_batch(texts)
    p.write_text(n.to_json())
    assert load(p).encode_batch(texts) == t.encode_batch(texts)
    for text in texts:
        assert n.decode(n.encode(text)) == text
