class SentencePieceTokenizer:
    def __init__(self) -> None:
        pass
        
    """Description: Trains the tokenizer"""
    def train(self, text: str, vocabSize: int) -> None:
        pass

    """Description: Encodes input text to a compressed sequence of integers """
    def encoder(self, text) -> list[int]:
        pass

    """Description: Inverse of Encoder -> Converts encoded text into human-readable text input text """
    def decoder(self, ids: list[int]) -> str:
        pass

    ########################################################
    ################### HELPER FUNCTIONS ###################
    ########################################################

if __name__ == "__main__":
    SentencePieceTokenizerInstance = SentencePieceTokenizer()
    filePath = "data/taylorSwift.txt"
    with open(filePath, 'r', encoding='utf-8') as file:
        fileContent = file.read()
    sampleText = "Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to “support Unicode” in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. I don’t blame programmers for still finding the whole thing mysterious, even 30 years after Unicode’s inception."
    vocabSize = 276

    print("\n" + sampleText + "\n")
    SentencePieceTokenizerInstance.train(sampleText, vocabSize)
    listOfEncodedIntegers = SentencePieceTokenizerInstance.encoder(fileContent)
    assert(len(listOfEncodedIntegers) > 0)
    decodedText = SentencePieceTokenizerInstance.decoder(listOfEncodedIntegers)
    assert(decodedText != "")
    assert(fileContent == decodedText) # text == decoder(encoder(text))

