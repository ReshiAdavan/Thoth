# Makefile

# Define the Python interpreter
PYTHON = python

# Define the paths to the Python scripts
REGEX_TOKENIZER_PATH = src/regexTokenizer.py
GPT4_TOKENIZER_PATH = src/gpt4Tokenizer.py
LLAMA2_TOKENIZER_PATH =	src/llama2Tokenizer.py
BASIC_TOKENIZER_PATH = src/basicTokenizer.py

# Default target
all: run_basic_tokenizer

# Target to run regexTokenizer.py
run_regex_tokenizer:
	$(PYTHON) $(REGEX_TOKENIZER_PATH)

# Target to run gpt4Tokenizer.py
run_gpt4_tokenizer:
	$(PYTHON) $(GPT4_TOKENIZER_PATH)

# Target to run gpt4Tokenizer.py
run_llama2_tokenizer:
	$(PYTHON) $(LLAMA2_TOKENIZER_PATH)

# Target to run gpt4Tokenizer.py
run_basic_tokenizer:
	$(PYTHON) $(BASIC_TOKENIZER_PATH)

# Phony target to avoid conflicts with files named 'clean'
.PHONY: all run_regex_tokenizer run_gpt4_tokenizer