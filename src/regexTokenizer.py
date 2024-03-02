import regex

GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class BasicTokenizer:
    def __init__(self) -> None:
        pass

    def train(self, sampleText: str) -> None:
        listOfEncodedIntegers = BasicTokenizerInstance.encoder(sampleText)
        decodedText = BasicTokenizerInstance.decoder(listOfEncodedIntegers)
        assert(sampleText == decodedText) # text == decoder(encoder(text))

    def encoder(self, text: str) -> list[int]:
        if len(text) == 0:
            raise ValueError("[Thoth => encoder]: String empty. Nothing to encode")
        print("[Thoth => encoder]: Encoding text given...")

        self.mints = []
        self.minter_idx = 256

        tokens = self.tokenize(text)
        encodedTokens = self.encodeTokens(tokens)
        return encodedTokens
    
    def decoder(self, ids: list[int]) -> str:
        if len(ids) == 0:
            raise ValueError("[Thoth => decoder]: No IDs. Nothing to decode.")
        
        print("[Thoth => decoder]: Decoding encoding given...")


    ########################################################
    ################### HELPER FUNCTIONS ###################
    ########################################################

    def tokenize(self, text_) -> list[str]:
        self.pattern = regex.compile(GPT4_SPLIT_PATTERN)
        return regex.findall(GPT4_SPLIT_PATTERN, text_)
    
    def encodeTokens(self, tokens) -> list[int]:
        ids = []
        for token in tokens:
            tokenUTF8 = token.encode("utf-8")
            tokenIds = self.mint(tokenUTF8)
            ids.extend(tokenIds)
        return ids

    def mint(self, partition) -> list[int]:
        partitionIds = list(partition)
        while len(partitionIds) >= 2:
            sortedCommonTuples = self.countCommonEncodedTuples(partitionIds)
            pair = min(sortedCommonTuples, key=lambda p: self.merges.get(p, float("inf")))
            print(pair)
    
    def countCommonEncodedTuples(self, encodedInts: list[int]) -> list[tuple]:
        freqDict = {}
        for i in range(len(encodedInts) - 1):
            pair = (encodedInts[i], encodedInts[i + 1])
            freqDict[pair] = freqDict.get(pair, 0) + 1 
        # {} [K, V] -> [(encodedInt1, encodedInt2), frequency]
        return sorted(((v, k) for k, v in freqDict.items()))




if __name__ == "__main__":
    BasicTokenizerInstance = BasicTokenizer()
    text = "Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to “support Unicode” in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. I don’t blame programmers for still finding the whole thing mysterious, even 30 years after Unicode’s inception."
    BasicTokenizerInstance.train(text)
