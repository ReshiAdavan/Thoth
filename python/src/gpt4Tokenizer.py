"""Reference-compatible named tiktoken encodings; ordinary text by default."""
import tiktoken

class GPT4Tokenizer:
    def __init__(self, encoding="cl100k_base"):
        self.encoding_name = encoding
        self.encoding = tiktoken.get_encoding(encoding)

    def encode(self, text, allowed_special=frozenset()):
        return self.encoding.encode(text, allowed_special=allowed_special, disallowed_special=())

    def decode(self, ids, errors="strict"):
        return self.encoding.decode(ids, errors=errors)

    def decode_bytes(self, ids):
        return self.encoding.decode_bytes(ids)

    def encode_batch(self, texts):
        return self.encoding.encode_batch(texts, disallowed_special=())

    encoder = encode
    decoder = decode
