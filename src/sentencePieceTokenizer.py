import unicodedata
class SentencePieceTokenizer:
    def __init__(self) -> None:
            self.vocab = {} # Initialize the vocabulary
            self.tokenCounter = len(self.vocab) # Start token counter after initial vocabulary

    def train(self, text: str, vocabSize: int) -> None:
        """Description: Trains the tokenizer"""
        assert (vocabSize >= 256)
        if len(text) == 0:
            raise ValueError("[Thoth => train]: String empty. Nothing to train on.")

        print("[Thoth => train]: Training...")

        # Convert the text to a list of Unicode code points
        codePoints = [ord(char) for char in text]
        self.tokenCounter = max(codePoints)
        # Initialize the vocabulary with the most common code points
        self.vocab = self.mostCommonCodepoints(codePoints)
        # Initialize the BPE merges
        self.merges = {}
        numOfMerges = vocabSize - 256

        # Perform BPE training
        for _ in range(numOfMerges):
            # Count the frequency of each pair of code points
            pairCounts = self.countPairFrequencies(codePoints)
            # Find the most frequent pair of code points
            mostFrequentPair = max(pairCounts, key=pairCounts.get)
            # Merge the most frequent pair into a new token
            newToken = self.generateNewToken()
            # Update the vocabulary and the list of code points
            self.vocab[newToken] = mostFrequentPair
            codePoints = self.mergeItemsReplace(codePoints, mostFrequentPair, newToken)

        # Store the final vocabulary and merges
        self.vocab = {token: self.vocab[token] for token in self.vocab if isinstance(token, int)}
        self.merges = {pair: token for token, pair in self.vocab.items() if isinstance(pair, tuple)}
        
        print("[Thoth => train]: Training complete.")

    def encoder(self, text: str) -> list[int]:
        """Description: Encodes input text to a compressed sequence of integers"""

        print("[Thoth => encoder]: Encoding...")
        # Convert the text to a list of Unicode code points
        codePoints = [ord(char) for char in text]

        # Encode the text using the trained BPE model
        encoded = []
        i = 0
        while i < len(codePoints):
            # Start with the longest possible sequence
            sequenceLength = min(len(codePoints) - i, max((len(k) if isinstance(k, tuple) else 1 for k in self.vocab.keys())))
            while sequenceLength > 0:
                sequence = tuple(codePoints[i:i+sequenceLength])
                if sequence in self.vocab:
                    encoded.append(self.vocab[sequence])
                    i += sequenceLength
                    break
                else:
                    # If no sequence is found, use the first code point as a fallback
                    encoded.append(codePoints[i])
                    i += 1
                sequenceLength -= 1

        print("[Thoth => encoder]: Encoding Complete...")
        return encoded

    def decoder(self, ids: list[int]) -> str:
        """Description: Inverse of Encoder -> Converts encoded text into human-readable text input text """

        print("[Thoth => decoder]: Decoding...")

        # Decode the encoded sequence using the trained BPE model
        decoded = []
        for token in ids:
            # Retrieve the sequence of code points for the token
            sequence = self.vocab.get(token, (token,))
            # Ensure that the sequence is a tuple of integers
            if isinstance(sequence, str):
                sequence = tuple(ord(char) for char in sequence)
            # Convert the sequence of code points back to a string
            decoded.extend(sequence)

        print("[Thoth => decoder]: Decoding Complete...")
        return ''.join(chr(codePoint) for codePoint in decoded)

    ########################################################
    ################### HELPER FUNCTIONS ###################
    ########################################################

    def countPairFrequencies(self, codePoints):
        """Counts the frequency of each pair of consecutive Unicode code points."""
        pairCounts = {}
        for i in range(len(codePoints) - 1):
            pair = (codePoints[i], codePoints[i + 1])
            if pair in pairCounts:
                pairCounts[pair] += 1
            else:
                pairCounts[pair] = 1
        return pairCounts

    def mergeItemsReplace(self, items, pair, newItem):
        """Merges a pair of consecutive items in a list with a new item."""
        mergedItems = []
        i = 0
        while i < len(items):
            if i < len(items) - 1 and items[i:i+2] == pair:
                mergedItems.append(newItem)
                i += 2
            else:
                mergedItems.append(items[i])
                i += 1
        return mergedItems
    
    def generateNewToken(self):
        """Generates a new token for a merged pair of Unicode code points."""
        newToken = self.tokenCounter
        self.tokenCounter += 1
        return newToken
    
    def mostCommonCodepoints(self, codepoints, topN = 256):
        """Returns a sorted dictionary of the most common codepoints"""
        from collections import Counter
        # Count occurrences of each codepoint
        counts = Counter(codepoints)
        # Sort by count and then by codepoint (to break ties)
        sortedCounts = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
        # Select the top N most common codepoints
        topCodepoints = sortedCounts[:topN]
        # Convert to a dictionary mapping codepoints to characters
        topCodepointsDict = {codepoint: chr(codepoint) for codepoint, count in topCodepoints} # K: codepoint int, V: codepoint char 
        return topCodepointsDict

if __name__ == "__main__":
    SentencePieceTokenizerInstance = SentencePieceTokenizer()
    filePath = "data/taylorSwift.txt"
    with open(filePath, 'r', encoding='utf-8') as file:
        fileContent = file.read()
    sampleText = "Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to “support Unicode” in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. I don’t blame programmers for still finding the whole thing mysterious, even 30 years after Unicode’s inception."
    vocabSize = 276

    # print("\n" + sampleText + "\n")
    SentencePieceTokenizerInstance.train(sampleText, vocabSize)
    listOfEncodedIntegers = SentencePieceTokenizerInstance.encoder(fileContent)
    assert(len(listOfEncodedIntegers) > 0)
    decodedText = SentencePieceTokenizerInstance.decoder(listOfEncodedIntegers)
    assert(decodedText != "")
    assert(fileContent == decodedText) # text == decoder(encoder(text))
