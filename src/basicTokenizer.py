class BasicTokenizer:
    def __init__(self) -> None:
        pass
        
    """Description: Trains the tokenizer"""
    def train(self, sampleText: str, vocabSize: int) -> None:
        assert(vocabSize >= 256)
        if len(sampleText) == 0:
            raise ValueError("[Thoth => train]: String empty. Nothing to train on.")

        print("[Thoth => train]: Training...")
        self.mints = {} # (int, int) -> int
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        
        tokens = text.encode('utf-8')
        encodedIntegers = [byte for byte in tokens]

        commonTuples = self.countCommonEncodedTuples(encodedIntegers)
        sortedCommonTuples = sorted(((v, k) for k, v in commonTuples.items()), reverse=True) 
        self.minter(sortedCommonTuples, encodedIntegers, 256, vocabSize)
        print("[Thoth => train]: Training complete.")

    """Description: Encodes input text to a compressed sequence of integers """
    def encoder(self, text: str) -> list[int]:
        if len(text) == 0:
            raise ValueError("[Thoth => encoder]: String empty. Nothing to encode.")
        
        print("[Thoth => encoder]: Encoding...")
        tokens = text.encode("utf-8")
        encodedIntegers = [byte for byte in tokens]

        while len(encodedIntegers) >= 2:
            commonTuples = self.countCommonEncodedTuples(encodedIntegers)
            pair = min(commonTuples, key=lambda p: self.mints.get(p, float("inf")))
            if pair not in self.mints:
                break
            idx = self.mints[pair]
            encodedIntegers = self.merge(pair, encodedIntegers, idx)
        print("[Thoth => encoder]: Encoding Complete...")
        return encodedIntegers

    """Description: Inverse of Encoder -> Converts encoded text into human-readable text input text """
    def decoder(self, ids: list[int]) -> str:
        if len(ids) == 0:
            raise ValueError("[Thoth => decoder]: No IDs. Nothing to decode.")
        
        print("[Thoth => decoder]: Decoding...")
        textBytes = b"".join(self.vocab[idx] for idx in ids)
        print("[Thoth => decoder]: Decoding Complete...")
        return textBytes.decode("utf-8", errors="replace")

    ########################################################
    ################### HELPER FUNCTIONS ###################
    ########################################################
    
    """Description: Returns most common tuples of encoded integers and their frequency of occurence"""
    def countCommonEncodedTuples(self, encodedInts: list[int]) -> list[tuple]:
        freqDict = {}
        for i in range(len(encodedInts) - 1):
            pair = (encodedInts[i], encodedInts[i + 1])
            freqDict[pair] = freqDict.get(pair, 0) + 1 
        # {} [K, V] -> [(encodedInt1, encodedInt2), frequency]
        return freqDict

    """Description: Replaces the most common tuple with a new integer token (created by the tokenizer)"""
    def mintToken(self, dictTuples: list[tuple], ids: list[int], idx: int) -> None:
        _, mostCommonPairOfInts  = dictTuples.pop(0)
        print(f"[Thoth => minter]: Minting {mostCommonPairOfInts} into a new token {idx}")
        self.mints[mostCommonPairOfInts] = idx
        self.vocab[idx] = self.vocab[mostCommonPairOfInts[0]] + self.vocab[mostCommonPairOfInts[1]]
        i = 0 
        while i < len(ids) - 1:
            if ids[i] == mostCommonPairOfInts[0] and ids[i + 1] == mostCommonPairOfInts[1]:
                ids[i:i + 2] = [idx]
                i -= 1
            i += 1
    
    """Description: Iteratively replaces common tuples with new integer tokens (minting)"""
    def minter(self, sortedDictTuples, idsList: list[int], start_index: int, end_index):
        while (start_index < end_index): # tuneable parameter
            self.mintToken(sortedDictTuples, idsList, start_index)
            start_index += 1

    """Description: Removes a tuple of integers from a list of integers"""
    def merge(self, pair, ids: list[int], idx: int) -> list[int]:
        mergedIDs, i = [], 0 

        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                mergedIDs.append(idx)
                i += 2 
            else:
                mergedIDs.append(ids[i])
                i += 1 
        return mergedIDs

if __name__ == "__main__":
    BasicTokenizerInstance = BasicTokenizer()
    filePath = "data/input/taylorSwift.txt"
    with open(filePath, 'r', encoding='utf-8') as file:
        fileContent = file.read()
    text = "Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to “support Unicode” in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. I don’t blame programmers for still finding the whole thing mysterious, even 30 years after Unicode’s inception."
    vocabSize = 276

    print("\n" + text + "\n")
    BasicTokenizerInstance.train(text, vocabSize)
    listOfEncodedIntegers = BasicTokenizerInstance.encoder(fileContent)
    assert(len(listOfEncodedIntegers) > 0)
    decodedText = BasicTokenizerInstance.decoder(listOfEncodedIntegers)
    assert(decodedText != "")
    assert(fileContent == decodedText) # text == decoder(encoder(text))