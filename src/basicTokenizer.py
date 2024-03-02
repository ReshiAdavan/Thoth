RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
MAGENTA = '\033[95m'
CYAN = '\033[96m'
RESET = '\033[0m'

class BasicTokenizer:
    def __init__(self) -> None:
        super().__init__()
    
    """Description: Trains the tokenizer"""
    def train(self, sampleText: str) -> None:
        listOfEncodedIntegers = BasicTokenizerInstance.encoder(sampleText)
        assert(len(listOfEncodedIntegers) > 0)
        decodedText = BasicTokenizerInstance.decoder(listOfEncodedIntegers)
        assert(decodedText != "")
        assert(sampleText == decodedText) # text == decoder(encoder(text))

        if len(decodedText) > 1000:
            filePath = "data/output/decodedTxt.txt"
            with open(filePath, 'w', encoding='utf-8') as file:
                file.write(decodedText)

        else: 
            print(f"{GREEN}[Thoth => train]: Decoded text: \n\n {RESET}{decodedText}\n")
        print(f"{GREEN}[Thoth => train]: Done. Exiting...{RESET}")
    
    """Description: Encodes input text to a compressed sequence of integers """
    def encoder(self, text: str) -> list[int]:
        if len(text) == 0:
            print(f"{RED}[Thoth => encoder]: String empty. Nothing to encode.{RESET}")
            return []
        
        # BPE Algorithm
        print(f"{MAGENTA}[Thoth => encoder]: Encoding text given...")
        minter_idx = 256
        tokens = text.encode('utf-8')
        encodedIntegers = [byte for byte in tokens]

        print("[Thoth => encoder]: Extracting most common characters...")
        sortedCommonTuples = self.countCommonEncodedTuples(encodedIntegers)

        self.mints = []
        print(f"[Thoth => encoder]: Minting common characters...{RESET}")
        compressedEncodedInts = self.minter(sortedCommonTuples, encodedIntegers, minter_idx)
        print(f"{MAGENTA}[Thoth => encoder]: Minting Compression Ratio: {len(tokens) / len(compressedEncodedInts):.3f}X")
    
        print(f"[Thoth => encoder]: Encoding complete.{RESET}")
        return compressedEncodedInts

    """Description: Inverse of Encoder -> Converts encoded text into human-readable text input text """
    def decoder(self, ids: list[int]) -> str:
        if len(ids) == 0:
            print(f"{RED}[Thoth => decoder]: No IDs. Nothing to decode.{RESET}")
            return ""
        
        print(f"{BLUE}[Thoth => decoder]: Decoding encoding given...")
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (x0, x1), idx in self.mints:
            vocab[idx] = vocab[x0] + vocab[x1]
        
        print("[Thoth => decoder]: Producing text...")
        text_bytes = b"".join(vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        
        print(f"[Thoth => decoder]: Decoding complete.{RESET}")
        return text

    ########################################################
    ################### HELPER FUNCTIONS ###################
    ########################################################
    
    """Description: Returns most common tuples of encoded integers and their frequency of occurence"""
    def countCommonEncodedTuples(self, encodedInts: list[int]):
        freqDict = {}
        for i in range(len(encodedInts) - 1):
            pair = (encodedInts[i], encodedInts[i + 1])
            freqDict[pair] = freqDict.get(pair, 0) + 1 
        # {} [K, V] -> [(encodedInt1, encodedInt2), frequency]
        return sorted(((v, k) for k, v in freqDict.items()), reverse=True)

    """Description: Replaces the most common tuple with a new integer token (created by the tokenizer)"""
    def mintToken(self, dictTuples, ids: list[int], idx: int) -> None:
        _, mostCommonPairOfInts  = dictTuples.pop(0)
        print(f"{CYAN}[Thoth => minter]: Minting {mostCommonPairOfInts} into a new token {idx}{RESET}")
        self.mints.append((mostCommonPairOfInts, idx))
        i = 0 
        while i < len(ids) - 1:
            if ids[i] == mostCommonPairOfInts[0] and ids[i + 1] == mostCommonPairOfInts[1]:
                ids[i:i + 2] = [idx]
                i -= 1
            i += 1
    
    """Description: Iteratively replaces common tuples with new integer tokens (minting)"""
    def minter(self, sortedDictTuples, idsList: list[int], index: int) -> list[int]:
        end_minter_idx = 277
        # while (len(sortedDictTuples) > 0): # full compression ratio
        while (index < end_minter_idx): # tuneable parameter
            self.mintToken(sortedDictTuples, idsList, index)
            index += 1 
        return idsList

if __name__ == "__main__":
    BasicTokenizerInstance = BasicTokenizer()
    filePath = "data/input/taylorSwift.txt"
    with open(filePath, 'r', encoding='utf-8') as file:
        fileContent = file.read()
    # text = "Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to “support Unicode” in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. I don’t blame programmers for still finding the whole thing mysterious, even 30 years after Unicode’s inception."
    BasicTokenizerInstance.train(fileContent)
