import pytest
import tiktoken
from gpt4Tokenizer import GPT4Tokenizer
from llama2Tokenizer import Llama2Tokenizer

@pytest.mark.parametrize('encoding',['cl100k_base','o200k_base','p50k_base','r50k_base'])
def test_reference(encoding):
    t=GPT4Tokenizer(encoding)
    for text in ['', 'Hello 🦉 中文\n', '<|endoftext|>', '  def x():\n\tpass']:
        assert t.encode(text)==tiktoken.get_encoding(encoding).encode_ordinary(text)
        assert t.decode(t.encode(text))==text

def test_sentencepiece(tmp_path):
    t=Llama2Tokenizer()
    with pytest.raises(ValueError): t.encode('hello')
    t.train('hello hello world',256)
    assert t.decode(t.encode('hello world'))=='hello world'
    path=tmp_path/'tokenizer.model';path.write_bytes(t.processor.serialized_model_proto())
    assert Llama2Tokenizer(path).encode('hello world')==t.encode('hello world')
