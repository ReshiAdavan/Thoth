import io
from pathlib import Path
import pytest
from hypothesis import given, settings, strategies as st
from llama2Tokenizer import Llama2Tokenizer

native = pytest.importorskip("thoth_native")
Native = native.NativeLlama2Tokenizer
MODEL = Path("models/thoth-sentencepiece-en-code-v1.model")
python = Llama2Tokenizer(MODEL)
rust = Native(str(MODEL))


@given(st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=200))
@settings(max_examples=200)
def test_saved_model_parity(text):
    assert rust.encode(text, lossless=False) == python.processor.encode(text)
    ids = rust.encode(text)
    assert ids == python.encode(text)
    assert rust.decode(ids) == text


def test_batch_save_reload_and_special_tokens(tmp_path):
    texts = ["", "Hello 🦉 中文", "a▁b", " \r\n\0\t", "<s>"]
    assert rust.encode_batch(texts) == python.encode_batch(texts)
    path = tmp_path / "copy.model"
    rust.save(str(path))
    assert path.read_bytes() == MODEL.read_bytes()
    reloaded = Native.from_bytes(rust.to_bytes())
    assert reloaded.encode_batch(texts) == rust.encode_batch(texts)
    ids = rust.encode("hello", add_bos=True, add_eos=True)
    assert ids == [python.processor.bos_id()] + python.encode("hello") + [
        python.processor.eos_id()
    ]
    assert rust.decode(ids) == "hello"
    assert rust.vocab_size() == python.processor.get_piece_size()


@pytest.mark.parametrize("ids", [[True], [1.0], [-1], ["1"], [None], [1000000]])
def test_invalid_ids(ids):
    with pytest.raises(ValueError):
        rust.decode(ids)


@pytest.mark.parametrize(
    "model", [b"", b"not protobuf", b"\xff\xff", b"x" * (16 * 1024 * 1024 + 1)]
)
def test_invalid_models(model):
    with pytest.raises(ValueError):
        Native.from_bytes(model)


@pytest.mark.parametrize("kind", ["bpe", "unigram"])
@pytest.mark.parametrize("byte_fallback", [False, True])
def test_normalized_external_models(kind, byte_fallback):
    import sentencepiece as spm

    output = io.BytesIO()
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(["hello world café example text"] * 10),
        model_writer=output,
        model_type=kind,
        vocab_size=320 if byte_fallback else 32,
        byte_fallback=byte_fallback,
        hard_vocab_limit=False,
    )
    model = output.getvalue()
    p = spm.SentencePieceProcessor(model_proto=model)
    r = Native.from_bytes(model)
    for text in ["hello world", "  hello  ", "Ｈｅｌｌｏ", "ﬁ café", "🦉", "", "▁"]:
        ids = r.encode(text, lossless=False)
        assert ids == p.encode(text)
        assert r.decode(ids) == p.decode(ids)
        if p.decode(ids) != text and not byte_fallback:
            with pytest.raises(ValueError, match="preserve"):
                r.encode(text)


@pytest.mark.parametrize(
    "corpus",
    [
        "hello hello world",
        "abababab aaaaaaaa abcabcabc",
        "你好 你好 café café\n🦉 e\u0301 🦉",
        "  spaces   and\ttabs\r\nnew line",
        "<unk> <s> </s> <0x41> ▁",
    ],
)
def test_native_training_interoperability(corpus, tmp_path):
    import sentencepiece as spm

    r = Native.train(corpus, 320)
    assert r.to_bytes() == Native.train(corpus, 320).to_bytes()
    assert r.vocab_size() <= 320
    p = spm.SentencePieceProcessor(model_proto=r.to_bytes())
    for text in [corpus, "unseen 中文 🦉\n\0", "a▁b", " e\u0301\r\n", ""]:
        canonical = r.encode(text, lossless=False)
        assert canonical == p.encode(text)
        assert r.decode(canonical) == p.decode(canonical)
        assert r.decode(r.encode(text)) == text
    path = tmp_path / "rust-trained.model"
    r.save(str(path))
    py = Llama2Tokenizer(path)
    assert py.encode_batch([corpus, "a▁b", "unseen 🦉"]) == r.encode_batch(
        [corpus, "a▁b", "unseen 🦉"]
    )


@pytest.mark.parametrize(
    "text,size",
    [
        ("", 300),
        ("hello", 256),
        ("hello", 259),
        ("\n", 300),
        ("a\0b", 300),
        ("hello", 65537),
        ("a" * (2 * 1024 * 1024 + 1), 300),
    ],
)
def test_native_training_bounds(text, size):
    with pytest.raises(ValueError):
        Native.train(text, size)


@given(st.text(alphabet="aaabbc hello world 你好🦉", min_size=1, max_size=150))
@settings(max_examples=100)
def test_native_trained_model_randomized_oracle(corpus):
    import sentencepiece as spm

    r = Native.train(corpus, 320)
    p = spm.SentencePieceProcessor(model_proto=r.to_bytes())
    probes = [corpus, corpus[::-1], corpus + corpus, "a▁b", "hello world"]
    for text in probes:
        ids = r.encode(text, lossless=False)
        assert ids == p.encode(text)
        assert r.decode(ids) == p.decode(ids)
        assert r.decode(r.encode(text)) == text


@given(st.text(alphabet="aaabcde fghijklmnopqrstuvwxyz01234\n\r\t▁🦉", max_size=500))
@settings(max_examples=200)
def test_saved_model_merge_heavy_parity(text):
    assert rust.encode(text, lossless=False) == python.processor.encode(text)
