# Thoth

Named after the Egyptian god of the moon, of reckoning, of learning, and of writing, is a **_tokenizer_**, purposed for interpreting text input for large-scale language models, such as OpenAI's GPT series of models (e.g., GPT-3.5, GPT-4, GitHub Copilot), Google's PaLM and Gemini, Meta's LLaMA family, and Anthropic's Claude models.

### Inspiration

I just felt like it. It seems like the bottleneck of LLMs at the moment, so it would be good to research and develop one, understand up-to-date architectures, and produce better open source implementations in hopes to provide AI at NLP.

### How to Use

Run each of the python files.

`python -u <directory of python file>`

Example: `python -u "c:\Users\...\gpt4Tokenizer.py"`

### Architecture

Differs per tokenizer.

#### BasicTokenizer

The `BasicTokenizer` class in `basicTokenizer.py` is designed to perform tokenization tasks, which are crucial for preparing text data for processing by large-scale language models. The architecture of `BasicTokenizer` is as follows:

##### Class Definition

- **BasicTokenizer**: The main class that encapsulates the tokenization process.

##### Methods

- **`__init__`**: Initializes the tokenizer instance.

- **`train`**: Trains the tokenizer on a given text dataset. It asserts that the vocabulary size is at least 256. It initializes the vocabulary and trains the tokenizer by identifying common byte sequences and replacing them with new tokens.

- **`encoder`**: Encodes input text into a compressed sequence of integers. It processes the text by encoding it into bytes, identifying common byte sequences, and replacing them with new tokens.

- **`decoder`**: Decodes the encoded sequence of integers back into human-readable text. It reverses the encoding process by replacing tokens with their corresponding byte sequences.

- **`countCommonEncodedTuples`**: Helper function that counts the frequency of common byte sequences in the encoded text.

- **`mintToken`**: Helper function that replaces the most common byte sequence with a new token.

- **`minter`**: Helper function that iteratively replaces common byte sequences with new tokens until the desired vocabulary size is reached.

##### Key Components

- **Vocabulary (`self.vocab`)**: A dictionary that maps integer tokens to their corresponding byte sequences.

- **Minters (`self.mints`)**: A dictionary that maps byte sequences to their corresponding integer tokens.

##### Usage

1. Initialize a `BasicTokenizer` instance.
2. Train the tokenizer on a text dataset using the `train` method.
3. Encode text into a compressed sequence of integers using the `encoder` method.
4. Decode the encoded sequence back into human-readable text using the `decoder` method.

This architecture allows for efficient text encoding and decoding, making it suitable for applications involving large-scale language models.

#### RegexTokenizer

The `RegexTokenizer` class in `regexTokenizer.py` is designed to perform tokenization tasks using regular expressions, providing a flexible and powerful method for text processing. The architecture of `RegexTokenizer` is as follows:

##### Class Definition

- **RegexTokenizer**: The main class that encapsulates the tokenization process using regular expressions.

##### Methods

- **`__init__`**: Initializes the tokenizer instance.

- **`train`**: Trains the tokenizer on a given text dataset. It asserts that the vocabulary size is at least 256. It initializes the vocabulary and trains the tokenizer by identifying common byte sequences and replacing them with new tokens.

- **`encoder`**: Encodes input text into a compressed sequence of integers. It processes the text by encoding it into bytes, identifying common byte sequences, and replacing them with new tokens.

- **`decoder`**: Decodes the encoded sequence of integers back into human-readable text. It reverses the encoding process by replacing tokens with their corresponding byte sequences.

- **`chunkify`**: Helper function that processes the encoded bytes to identify common byte sequences and replaces them with new tokens.

##### Key Components

- **Vocabulary (`self.vocab`)**: A dictionary that maps integer tokens to their corresponding byte sequences.

- **Minters (`self.mints`)**: A dictionary that maps byte sequences to their corresponding integer tokens.

##### Usage

1. Initialize a `RegexTokenizer` instance.
2. Train the tokenizer on a text dataset using the `train` method.
3. Encode text into a compressed sequence of integers using the `encoder` method.
4. Decode the encoded sequence back into human-readable text using the `decoder` method.

This architecture allows for efficient text encoding and decoding using regular expressions, making it suitable for applications involving large-scale language models.

#### GPT4Tokenizer

The `GPT4Tokenizer` class in `gpt4Tokenizer.py` is designed to perform tokenization tasks specifically tailored for models like GPT-4, ensuring compatibility and efficiency in text processing. The architecture of `GPT4Tokenizer` is as follows:

##### Class Definition

- **GPT4Tokenizer**: The main class that encapsulates the tokenization process, optimized for GPT-4's requirements.

##### Methods

- **`__init__`**: Initializes the tokenizer instance, setting up the tokenization rules and vocabulary.

- **`tokenize`**: Tokenizes input text according to GPT-4's tokenization rules. It processes the text by applying the tokenization logic, which may include splitting text into subwords, handling special characters, and more.

- **`detokenize`**: Converts a sequence of tokens back into human-readable text. It reverses the tokenization process, reconstructing the original text from the tokenized sequence.

- **`encode`**: Encodes input text into a sequence of integers, representing the tokens in a format that GPT-4 can process.

- **`decode`**: Decodes a sequence of integers back into human-readable text, reversing the encoding process.

##### Key Components

- **Vocabulary (`self.vocab`)**: A dictionary that maps tokens to their corresponding integer IDs.

- **Tokenization Rules**: A set of rules for splitting text into tokens, handling special characters, and more, tailored to GPT-4's requirements.

##### Usage

1. Initialize a `GPT4Tokenizer` instance.
2. Tokenize text using the `tokenize` method, which applies GPT-4's tokenization rules to the input text and returns a list of tokens.
3. Encode text into a sequence of integers using the `encode` method, preparing the text for input into GPT-4.
4. Decode the encoded sequence back into human-readable text using the `decode` method.

This architecture is specifically designed for tokenization tasks with GPT-4, ensuring compatibility and efficiency in text processing for this model.

#### SentencePiece

- TBD

#### Llama2

- TBD

### Topics

- **Languages**: Python
- **Libraries/Frameworks/Tools**: Regex, Tiktoken, Unicodedata
- <ins>**Other**</ins>:
  - **Concepts**: Tokenization, Embeddings, LLMs, GPT2.0 & GPT4.0, Llama2, Sentencepiece, Byte-pair encoding (BPE), ...
