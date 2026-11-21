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


@pytest.mark.parametrize("ids", [[True], [65.0], [-1], ["65"], [None], [1000000]])
def test_native_invalid_ids(ids):
    with pytest.raises(ValueError):
        native.NativeTokenizer().decode(ids)


from hypothesis import given, settings, strategies as st
import json


@given(
    st.text(alphabet="abc  \n🦉é", min_size=1, max_size=150),
    st.integers(min_value=256, max_value=300),
)
@settings(max_examples=100)
def test_native_training_parity(text, size):
    p = BasicTokenizer()
    n = native.NativeTokenizer()
    p.train(text, size)
    n.train(text, size)
    assert list(p.mints) == [tuple(pair) for pair in json.loads(n.to_json())["merges"]]
    probe = text + " unseen 中文\r\n"
    assert p.encode(probe) == n.encode(probe)
    assert n.decode(n.encode(probe)) == probe
