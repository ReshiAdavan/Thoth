# Thoth

Named after the Egyptian god of the moon, of reckoning, of learning, and of writing, is a **_tokenizer_**, purposed for interpreting text input for large-scale language models, such as OpenAI's GPT series of models (e.g., GPT-3.5, GPT-4, GitHub Copilot), Google's PaLM and Gemini, Meta's LLaMA family, and Anthropic's Claude models.

### Inspiration

I just felt like it. It seems like the bottleneck of LLMs at the moment, so it would be good to research and develop one, understand up-to-date architectures, and produce better open source implementations in hopes to provide AI in NLP.

### How to Use

#### Python

Navigate to the `python/` directory and run the `Makefile`.

Example: `cd python/ ; make`

And you can change which tokenizer you want to run by configuring the `Makefile`

#### Rust

Navigate to the `rust` directory and run `cargo build` and then `cargo run`

Example: `cd rust/ ; cargo build ; cargo run`

And you can change which tokenizer you want to run by specifying in `main.rs`

### Architecture 

#### Python

Differs per tokenizer.

##### BasicTokenizer

This tokenizer designed to perform tokenization by partitioning input text data by collection of bytes. It does not particularly consider the type of text data it analyzes.

###### Methods

- **`train`**: Trains the tokenizer on a given text dataset. Initializes vocabulary and trains by identifying common byte sequences and replacing them with new tokens.
- **`encoder`**: Encodes input text into a compressed sequence of integers. It processes the text by encoding it into bytes, identifying common byte sequences, and replacing them with new tokens.
- **`decoder`**: Decodes the encoded sequence of integers back into human-readable text. It reverses the encoding process by replacing tokens with their corresponding byte sequences.

###### Key Components

- **Vocabulary (`self.vocab`)**: A dictionary that maps integer tokens to their corresponding byte sequences.
- **Minters (`self.mints`)**: A dictionary that maps byte sequences to their corresponding integer tokens.

##### RegexTokenizer

This tokenizer is designed to perform tokenization tasks using regular expressions. RegeX at least can be made to have some general understanding of the relationships within text data.

###### Methods

- **`train`**: Trains the tokenizer on a given text dataset. Initializes the vocabulary and trains the tokenizer by identifying sequences through RegeX experessions and replaces them with new tokens.
- **`encoder`**: Encodes input text into a compressed sequence of integers. It processes the through RegeX, identifying common byte sequences, and replacing them with new tokens.
- **`decoder`**: Decodes the encoded sequence of integers back into human-readable text. It reverses the encoding process by replacing tokens with their corresponding representations.

###### Key Components

- **Vocabulary (`self.vocab`)**: A dictionary that maps integer tokens to their corresponding sequences.
- **Minters (`self.mints`)**: A dictionary that maps sequences to their corresponding integer tokens.

##### GPT4Tokenizer

This tokenizer is designed to perform tokenization tasks specifically tailored for models like GPT-4, ensuring compatibility and efficiency in text processing.

###### Methods

- **`tokenize`**: Tokenizes input text according to GPT-4's tokenization rules. It processes the text by applying the tokenization logic, which may include splitting text into subwords, handling special characters, and more.
- **`detokenize`**: Converts a sequence of tokens back into human-readable text. It reverses the tokenization process, reconstructing the original text from the tokenized sequence.
- **`encode`**: Encodes input text into a sequence of integers, representing the tokens in a format that GPT-4 can process.
- **`decode`**: Decodes a sequence of integers back into human-readable text, reversing the encoding process.

###### Key Components

- **Vocabulary (`self.vocab`)**: A dictionary that maps tokens to their corresponding integer IDs.
- **Tokenization Rules**: A set of rules for splitting text into tokens, handling special characters, and more, tailored to GPT-4's requirements.

##### Llama2Tokenizer

This tokenizer is designed to perform tokenization tasks using the SentencePiece algorithm, which is a subword tokenization method that is particularly effective for languages with rich morphology. This class encapsulates the process of training a tokenizer on a given text dataset, encoding input text into a compressed sequence of integers, and decoding the encoded sequence back into human-readable text.

###### Methods

- **`train`**: Trains the tokenizer on a given text dataset by identifying the most frequent pairs of Unicode code points and merging them into new tokens, expanding the vocabulary to the specified size.
- **`encoder`**: Encodes input text into a compressed sequence of integers by breaking down the text into the most frequent subword units identified during training.
- **`decoder`**: Decodes the encoded sequence of integers back into human-readable text by reversing the encoding process.

###### Key Components

- **Vocabulary (`self.vocab`)**: A dictionary that maps integer tokens to their corresponding Unicode code points or merged pairs of code points.
- **Token Counter (`self.tokenCounter`)**: A counter used to assign unique identifiers to new tokens generated during the training process.

#### Rust

Differs per tokenizer.

##### BasicTokenizer

This tokenizer is designed to perform tokenization by partitioning input text data into collections of bytes and encoding them into sequences of integer tokens. It does not particularly consider the type of text data it analyzes.

###### Methods

- **`train`**: Trains the tokenizer on a given text dataset. Initializes the vocabulary and identifies common byte sequences, replacing them with new tokens until the desired vocabulary size is reached.
- **`encoder`**: Encodes input text into a compressed sequence of integers. It processes the text by converting it to bytes, identifying common byte sequences, and replacing them with new tokens based on the previously trained vocabulary.
- **`decoder`**: Decodes the encoded sequence of integers back into human-readable text. It reverses the encoding process by replacing tokens with their corresponding byte sequences from the vocabulary.

###### Key Components

- **Vocabulary (`self.vocab`)**: An index map that maps integer tokens to their corresponding byte sequences.
- **Minters (`self.mints`)**: A hashmap that maps byte sequence pairs to their corresponding integer tokens.

##### Llama2Tokenizer

WIP

### Topics

- **Languages**: Python, Rust
- **Libraries/Frameworks/Tools**: Regex, Tiktoken, Unicodedata
- <ins>**Other**</ins>:
  - **Concepts**: Tokenization, Embeddings, LLMs, GPT2.0 & GPT4.0, Llama2, Sentencepiece, Byte-pair encoding (BPE)
