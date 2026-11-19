import pytest
from hypothesis import given, strategies as st
from basicTokenizer import BasicTokenizer
from regexTokenizer import RegexTokenizer

@pytest.mark.parametrize("cls", [BasicTokenizer, RegexTokenizer])
def test_training(cls):
    t = cls()
    t.train("abababab", 270)
    assert len(t.vocab) <= 270
    assert t.decode(t.encode("ab 🦉 中文\n")) == "ab 🦉 中文\n"
    t.train("a", 300)
    assert len(t.vocab) == 256
    assert t.encode("") == []
    assert t.decode([]) == ""
    with pytest.raises(ValueError): t.decode([999999])

def test_hierarchical_merges():
    t = BasicTokenizer()
    t.train("abababab", 258)
    assert t.mints == {(97,98):256, (256,256):257}
    assert t.encode("abababab") == [257,257]

@given(st.text(alphabet=st.characters(blacklist_categories=("Cs",))))
def test_roundtrip(text):
    for cls in (BasicTokenizer, RegexTokenizer):
        t = cls()
        t.train("hello world hello 🦉", 280)
        assert t.decode(t.encode(text)) == text
