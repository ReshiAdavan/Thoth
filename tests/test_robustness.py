import io
import pytest
from hypothesis import given, settings, strategies as st
from basicTokenizer import BasicTokenizer
from regexTokenizer import RegexTokenizer
from llama2Tokenizer import Llama2Tokenizer


@pytest.mark.parametrize(
    "factory",
    [
        BasicTokenizer,
        RegexTokenizer,
        lambda: Llama2Tokenizer("models/thoth-sentencepiece-en-code-v1.model"),
    ],
)
@pytest.mark.parametrize("ids", [[True], [65.0], [-1], ["65"], [None], [1000000]])
def test_invalid_ids(factory, ids):
    with pytest.raises(ValueError):
        factory().decode(ids)


@pytest.fixture(scope="module")
def trained_sp():
    t = Llama2Tokenizer()
    t.train("hello hello world", 300)
    return t


@pytest.mark.parametrize(
    "text",
    ["", "a▁b", "▁", "hello 🦉", "\r\n\t", "\x00", "e\u0301", "  hello  ", "<0x41>"],
)
def test_sentencepiece_preserves_edges(trained_sp, text, tmp_path):
    for t in [
        trained_sp,
        Llama2Tokenizer("models/thoth-sentencepiece-en-code-v1.model"),
    ]:
        assert t.decode(t.encode(text)) == text
        path = tmp_path / "model"
        t.save(path)
        reloaded = Llama2Tokenizer(path)
        assert reloaded.encode(text) == t.encode(text)


@given(st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=150))
@settings(max_examples=100)
def test_saved_sentencepiece_arbitrary_unicode(text):
    t = Llama2Tokenizer("models/thoth-sentencepiece-en-code-v1.model")
    assert t.decode(t.encode(text)) == text


def test_external_lossy_model_rejected():
    import sentencepiece as spm

    model = io.BytesIO()
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(["hello world"]),
        model_writer=model,
        vocab_size=32,
        model_type="bpe",
        hard_vocab_limit=False,
    )
    t = Llama2Tokenizer()
    t.processor = spm.SentencePieceProcessor(model_proto=model.getvalue())
    assert t.encode("hello") == t.processor.encode("hello")
    with pytest.raises(ValueError, match="preserve"):
        t.encode("🦉")


def test_training_rejects_small_vocabulary(trained_sp):
    before = trained_sp.encode("hello")
    with pytest.raises(ValueError):
        trained_sp.train("hello", 256)
    assert trained_sp.encode("hello") == before
