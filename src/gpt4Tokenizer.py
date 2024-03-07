"""Implements GPT-4 Tokenizer as wrapper around RegexTokenizer."""

import tiktoken
from regexTokenizer import RegexTokenizer

def bpe(mergeableRanks, token, maxRank):
    # Helper function used in getGpt4Merges() to reconstruct the merge forest
    parts = [bytes([b]) for b in token]
    while True:
        minIdx = None
        minRank = None
        for i, pair in enumerate(zip(parts[:-1], parts[1:])):
            rank = mergeableRanks.get(pair[0] + pair[1])
            if rank is not None and (minRank is None or rank < minRank):
                minIdx = i
                minRank = rank
        if minRank is None or (maxRank is not None and minRank >= maxRank):
            break
        assert minIdx is not None
        parts = parts[:minIdx] + [parts[minIdx] + parts[minIdx + 1]] + parts[minIdx + 2:]
    return parts

def recoverMerges(mergeableRanks):
    # the `merges` are already the byte sequences in their merged state 
    # so we have to recover the original pairings. We can do this by doing
    # a small BPE training run on all the tokens, in their order.
    # Refs: https://github.com/openai/tiktoken/issues/60 & https://github.com/karpathy/minbpe/issues/11#issuecomment-1950805306
    merges = {}
    for token, rank in mergeableRanks.items():
        if len(token) == 1:
            continue # Skip raw bytes
        pair = tuple(bpe(mergeableRanks, token, maxRank=rank))
        assert len(pair) == 2
        # Recover the integer ranks of the pair
        ix0 = mergeableRanks[pair[0]]
        ix1 = mergeableRanks[pair[1]]
        merges[(ix0, ix1)] = rank

    return merges

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
GPT4_SPECIAL_TOKENS = {
    '<|endoftext|>': 100257,
    '<|fim_prefix|>': 100258,
    '<|fim_middle|>': 100259,
    '<|fim_suffix|>': 100260,
    '<|endofprompt|>': 100276
}

class GPT4Tokenizer(RegexTokenizer):
    """Lightweight wrapper on RegexTokenizer that matches GPT-4's tokenizer."""

    def __init__(self):
        super().__init__(pattern=GPT4_SPLIT_PATTERN)
        # Get the official tokenizer and its merges
        enc = tiktoken.getEncoding("cl100k_base")
        mergeableRanks = enc._mergeableRanks
        # The merges are those of gpt4, but we have to recover them
        self.merges = recoverMerges(mergeableRanks)
        # Reconstruct the vocab from the merges
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        self.vocab = vocab
        # Now here is another tricky part.
        # for some reason, the tokens corresponding to individual bytes
        # are permuted in a different order. This is completely non-sensical
        # and probably historical, but therefore we have to deal with it here.
        self.byteShuffle = {i: mergeableRanks[bytes([i])] for i in range(256)}
        self.inverseByteShuffle = {v: k for k, v in self.byteShuffle.items()}
        # Finally register the special tokens
        self.registerSpecialTokens(GPT4_SPECIAL_TOKENS)

    def encodeChunk(self, textBytes):
        # Before we start processing bytes, we have to permute them
        textBytes = bytes(self.byteShuffle[b] for b in textBytes)
        ids = super().chunkify(textBytes)
        return ids

    def decode(self, ids):
        # We have to un-permute the bytes before we decode
        textBytes = b"".join(self.vocab[idx] for idx in ids)
        textBytes = bytes(self.inverseByteShuffle[b] for b in textBytes)
        text = textBytes.decode("utf-8", errors="replace")
        return text

    def saveVocab(self, vocabFile):
        # just for visualization purposes let's output the GPT-4 tokens
        # in the exact same format as the base class would.
        # simple run as:
        # from minbpe import GPT4Tokenizer; GPT4Tokenizer().save_vocab('gpt4.vocab')
        from .utils import renderToken
        # Build vocab being mindful of the byte shuffle
        vocab = {idx: bytes([self.inverseByteShuffle[idx]]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        # Now merge the shuffled bytes and write to file
        invertedMerges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocabFile, "w", encoding="utf-8") as f:
            for idx, token in vocab.items():
                s = renderToken(token)
                if idx in invertedMerges:
                    idx0, idx1 = invertedMerges[idx]
                    s0 = renderToken(vocab[idx0])
                    s1 = renderToken(vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    f.write(f"[{s}] {idx}\n")
