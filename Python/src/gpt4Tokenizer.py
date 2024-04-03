"""Implements GPT-4 Tokenizer as wrapper around RegexTokenizer."""

import tiktoken
from regexTokenizer import RegexTokenizer

def bpe(mergeableRanks, token, maxRank):
    """
    - Implements Byte Pair Encoding (BPE) algorithm to reconstruct the merge forest for a given token.
    - Used in `recoverMerges()` to reconstruct the merge forest.
    - The algorithm iteratively merges the most frequent pair of bytes in the token until no more merges can be made or the maximum rank is reached.
    - The merging process is guided by the `mergeableRanks` dictionary, which maps pairs of bytes to their merge ranks. 

    Parameters:
    - mergeableRanks (dict): A dictionary mapping pairs of bytes to their merge ranks.
    - token (bytes): The token to be processed.
    - maxRank (int, optional): The maximum rank to consider for merges. If not provided, the process continues until no more merges can be made.

    Returns:
    - list: A list of parts, where each part is a byte or a merged byte sequence.
    """
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
    """
    - Recovers original pairings of byte sequences that have been merged during the tokenization process.
    - Performs a small Byte Pair Encoding (BPE) training run on all the tokens, in their order, to recover the original pairings.
    - Iterates over each token and its rank in the `mergeableRanks {}`. For tokens that are not single bytes, it uses the `bpe` function to reconstruct the original pairings. 
    - Pairings are stored in `merges {}` -> K: tuple of the original byte sequences & V: rank of the merged sequence.

    Parameters:
    - mergeableRanks (dict): A dictionary mapping pairs of bytes to their merge ranks.

    Returns:
    - dict: A dictionary where each key is a tuple of the original byte sequences and the value is the rank of the merged sequence.
    """
    # merges are byte sequences in their merged state so recover the original pairings. 
    # Do a small BPE training run on all the tokens, in their order.
    # Refs: https://github.com/openai/tiktoken/issues/60 & https://github.com/karpathy/minbpe/issues/11#issuecomment-1950805306
    merges = {}
    for token, rank in mergeableRanks.items():
        if len(token) == 1:
            # Skip raw bytes
            continue
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
        try: 
            enc = tiktoken.getEncoding("cl100k_base")
        except Exception as e:
            print(f"[Thoth]: Failed to get encoding with error: {e}")
            return
        mergeableRanks = enc._mergeableRanks
        # Recover GPT4 merges
        self.merges = recoverMerges(mergeableRanks)
        # Reconstruct the vocab from the merges
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        self.vocab = vocab
        # The tokens corresponding to individual bytes are permuted in a different order. 
        self.byteShuffle = {i: mergeableRanks[bytes([i])] for i in range(256)}
        self.inverseByteShuffle = {v: k for k, v in self.byteShuffle.items()}
        # Register the special tokens
        self.registerSpecialTokens(GPT4_SPECIAL_TOKENS)

    def encodeChunk(self, textBytes):
        """
        - Encodes a chunk of text into a sequence of token IDs using the GPT-4 tokenizer.
        - Permutes bytes of input text before processing them.
        - Uses `byteShuffle` attribute of the `GPT4Tokenizer` class to reorder bytes according to a specific pattern. 
        - Calls `chunkify` method of the superclass (`RegexTokenizer`) to convert the permuted bytes into a sequence of token IDs.

        Parameters:
        - textBytes (bytes): The input text to be encoded, represented as bytes.

        Returns:
        - list: A list of token IDs representing the encoded input text.
        """
        # Permute bytes before processing them
        textBytes = bytes(self.byteShuffle[b] for b in textBytes)
        ids = super().chunkify(textBytes)
        return ids

    def decode(self, ids):
        """
        - Decodes a sequence of token IDs back into human-readable text using the GPT-4 tokenizer.
        - Reverses encoding process un-permuting bytes of the token IDs. 
        - Uses `inverseByteShuffle` attribute of the `GPT4Tokenizer` class to restore the original byte order. 
        - Joins bytes into a single byte sequence and decodes it into a string of text.

        Parameters:
        - ids (list): A list of token IDs representing the encoded input text.

        Returns:
        - str: The decoded text as a string.
        """

        # Un-permute the bytes before decoding
        textBytes = b"".join(self.vocab[idx] for idx in ids)
        textBytes = bytes(self.inverseByteShuffle[b] for b in textBytes)
        text = textBytes.decode("utf-8", errors="replace")
        return text

    def saveVocab(self, vocabFile):
        """
        - Saves the vocabulary used by the GPT-4 tokenizer to a file.
        - Builds the vocabulary, accounting byte shuffle. 
        - Merges shuffled bytes and writes them to the specified file. 
        - Vocabulary is saved in a format that includes the original byte sequences and their corresponding token IDs, 

        Parameters:
        - vocabFile (str): The path to the file where the vocabulary will be saved.

        Returns:
        - None
        """

        from .utils import renderToken
        # Build vocab being mindful of the byte shuffle
        vocab = {idx: bytes([self.inverseByteShuffle[idx]]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        # Merge the shuffled bytes and write to file
        invertedMerges = {idx: pair for pair, idx in self.merges.items()}
        try:
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
        except Exception as e:
            print(f"[Thoth]: An error occurred when writing to file: {e}")
