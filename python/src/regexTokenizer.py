import regex
from basicTokenizer import BasicTokenizer

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class RegexTokenizer(BasicTokenizer):
    def __init__(self, pattern=GPT4_SPLIT_PATTERN):
        super().__init__()
        self.pattern = pattern
        self.compiledPattern = regex.compile(pattern)

    def chunks(self, text):
        chunks = self.compiledPattern.findall(text)
        if any(not isinstance(chunk, str) for chunk in chunks) or "".join(chunks) != text:
            raise ValueError("Pattern must partition the entire input without capturing groups")
        return [chunk.encode("utf-8") for chunk in chunks]
