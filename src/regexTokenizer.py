import regex as re
import utils as util

GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class RegexTokenizer:
    def __init__(self) -> None:
        pass

    def train(self, text: str, vocabSize: int) -> None:
        assert (vocabSize >= 256)
        if len(text) == 0:
            raise ValueError("[Thoth => train]: String empty. Nothing to train on.")
        
        print("[Thoth => train]: Training...")

        idx = 256
        numOfMerges = vocabSize - 256
        self.compiledPattern = re.compile(GPT4_SPLIT_PATTERN)
        tokens = re.findall(self.compiledPattern, text)

        encodeIDsList = [list(token.encode("utf-8")) for token in tokens]

        self.mints = {} # (int, int) -> int
        self.vocab = {idx: bytes([idx]) for idx in range(256)} # idx -> bytes
        for i in range(numOfMerges):
            commonTuples = {}
            for encodeIDs in encodeIDsList:
                util.countCommonEncodedTuples(encodeIDs, commonTuples)
            pair = max(commonTuples, key=commonTuples.get)
            idx += i
            encodeIDsList = [util.merge(encodeIDs, pair, idx) for encodeIDs in encodeIDsList]
            self.mints[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

        print("[Thoth => train]: Training complete.")

    def encoder(self, text: str) -> list[int]:
        print("[Thoth => encoder]: Encoding...")
        tokens = re.findall(self.compiledPattern, text)
        encodedIntegers = []
        for token in tokens:
            tokenUTF8 = token.encode("utf-8") # raw bytes
            encodeIDs = self.chunkify(tokenUTF8)
            encodedIntegers.extend(encodeIDs)
        print("[Thoth => encoder]: Encoding Complete...")
        return encodedIntegers
    
    def decoder(self, encodedIntegers: list[int]) -> str:
        print("[Thoth => decoder]: Decoding...")
        bytes = []
        for idx in encodedIntegers:
            if idx in self.vocab:
                bytes.append(self.vocab[idx])
            else:
                raise ValueError(f"invalid token id: {idx}")
        joinedBytes = b"".join(bytes)
        print("[Thoth => decoder]: Decoding Complete...")
        return joinedBytes.decode("utf-8", errors="replace")

    ########################################################
    ################### HELPER FUNCTIONS ###################
    ########################################################

    def chunkify(self, joinedBytes):
        ids = list(joinedBytes)
        while len(ids) >= 2:
            commonTuples = util.countCommonEncodedTuples(ids)
            pair = min(commonTuples, key=lambda p: self.mints.get(p, float("inf")))
            if pair not in self.mints:
                break
            idx = self.merges[pair]
            ids = self.merge(ids, pair, idx)
        return ids

if __name__ == "__main__":
    RegexTokenizerInstance = RegexTokenizer()
    filePath = "data/taylorSwift.txt"
    with open(filePath, 'r', encoding='utf-8') as file:
        fileContent = file.read()
    sampleText = "Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to “support Unicode” in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. I don’t blame programmers for still finding the whole thing mysterious, even 30 years after Unicode’s inception."
    vocabSize = 276

    # print("\n" + sampleText + "\n")
    RegexTokenizerInstance.train(sampleText, vocabSize)
    listOfEncodedIntegers = RegexTokenizerInstance.encoder(fileContent)
    assert(len(listOfEncodedIntegers) > 0)
    decodedText = RegexTokenizerInstance.decoder(listOfEncodedIntegers)
    assert(decodedText != "")
    assert(fileContent == decodedText) # text == decoder(encoder(text))
