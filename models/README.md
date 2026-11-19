# Thoth English + Code v1

A **16,384-token regex byte-BPE tokenizer**, trained using Thoth's own merge algorithm. This is a tokenizer vocabulary and merge table, not neural-network weights and not a model-compatible replacement for an existing LLM tokenizer.

## Data and training

- 36,006,500 source bytes: approximately 25 MB of CPython 3.13 source/documentation and 11 MB of general English from NLTK's Gutenberg selections.
- Gutenberg training chunks have weight 3; CPython chunks have weight 1. Reweighting does not add unique training data.
- Complete files are held out before training: 226 CPython files selected deterministically by hashed path, plus Austen's *Persuasion* and Chesterton's *The Man Who Was Thursday*.
- Duplicate CPython file contents are removed. The 25 MB technical cap uses sorted file order, so this is not a uniformly sampled representation of all programming languages.
- General prose is predominantly historical literature, including the King James Bible. This first release is not representative of modern conversational English, all programming languages, or multilingual usage.
- An incremental pair-frequency trainer produces the same deterministic merge ordering as the original reference trainer on regression fixtures.

The model JSON stores the regex pattern and ordered byte-pair merges. The report includes every selected file, source hashes, split, sizes, evaluation counts and model checksum. Raw training corpora stay in ignored `training/cache/`.

## Held-out results

| Corpus | UTF-8 bytes | Thoth tokens | cl100k_base tokens |
|---|---:|---:|---:|
| Code and documentation | 5,370,665 | 1,387,382 | 1,295,555 |
| General prose | 786,817 | 199,793 | 184,948 |
| Total | 6,157,482 | 1,587,175 | 1,480,503 |

All 228 held-out files round-trip exactly after saving and reloading. Thoth uses approximately 7.2% more tokens than cl100k_base on this evaluation, with a substantially smaller vocabulary. That is a corpus-specific comparison, not evidence of equivalent LLM performance. The 16K target was fixed before the final training run; an earlier technical-only baseline was also evaluated on the CPython holdout.

## Use

Select **Thoth English + Code v1 · trained 16K** in the local playground, with `python -m thoth_demo` running. It loads the committed model without retraining. CLI:

```sh
printf 'Hello world!\n' | thoth encode --model models/thoth-code-en-v1.json
```

## Reproduce

From the repository root, activate the virtual environment and download:

```sh
mkdir -p training/cache
curl -L --fail https://github.com/python/cpython/archive/refs/tags/v3.13.0.tar.gz -o training/cache/cpython-v3.13.0.tar.gz
curl -L --fail https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/gutenberg.zip -o training/cache/gutenberg.zip
python training/train_release.py
```

Archive hashes are recorded and checked by the training script. No GPU is needed; byte-pair vocabulary training is a CPU counting/merging task.

## Attribution

CPython source/documentation is covered by the included [CPython license](CPython-LICENSE.txt). The tokenizer artifacts were generated from selected files and are not unmodified CPython source. NLTK identifies its Gutenberg selections as public domain: <https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/gutenberg.xml>. Source archive provenance and checksums are retained in the report.
