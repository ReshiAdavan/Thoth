"""Deterministic byte-pair encoding with a complete byte vocabulary."""

from collections import Counter


class BasicTokenizer:
    def __init__(self):
        self.mints = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

    def chunks(self, text):
        return [text.encode("utf-8")]

    @staticmethod
    def merge(ids, pair, token):
        result = []
        i = 0
        while i < len(ids):
            if i + 1 < len(ids) and tuple(ids[i : i + 2]) == pair:
                result.append(token)
                i += 2
            else:
                result.append(ids[i])
                i += 1
        return result

    def train(self, text, vocabSize):
        if type(vocabSize) is not int or vocabSize < 256:
            raise ValueError("Vocabulary size must be at least 256")
        if not text:
            raise ValueError("Training text must not be empty")
        self.mints = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        chunks = Counter(tuple(chunk) for chunk in self.chunks(text))
        for token in range(256, vocabSize):
            counts = Counter()
            for chunk, weight in chunks.items():
                for pair in zip(chunk, chunk[1:]):
                    counts[pair] += weight
            if not counts:
                break
            pair = min(counts, key=lambda pair: (-counts[pair], pair))
            self.mints[pair] = token
            self.vocab[token] = self.vocab[pair[0]] + self.vocab[pair[1]]
            updated = Counter()
            for chunk, weight in chunks.items():
                updated[tuple(self.merge(chunk, pair, token))] += weight
            chunks = updated

    def chunkify(self, chunk):
        ids = list(chunk)
        while len(ids) > 1:
            pairs = [p for p in zip(ids, ids[1:]) if p in self.mints]
            if not pairs:
                break
            pair = min(pairs, key=self.mints.get)
            ids = self.merge(ids, pair, self.mints[pair])
        return ids

    def encode(self, text):
        return [token for chunk in self.chunks(text) for token in self.chunkify(chunk)]

    def decode_bytes(self, ids):
        ids = list(ids)
        if any(type(token) is not int or token < 0 for token in ids):
            raise ValueError("Token IDs must be nonnegative integers")
        try:
            return b"".join(self.vocab[token] for token in ids)
        except KeyError as error:
            raise ValueError(f"Unknown token ID: {error.args[0]}") from error

    def decode(self, ids, errors="strict"):
        return self.decode_bytes(ids).decode("utf-8", errors=errors)

    def encode_batch(self, texts):
        return [self.encode(text) for text in texts]

    def encoder(self, text):
        return self.encode(text)

    def decoder(self, ids):
        return self.decode(ids)
