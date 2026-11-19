"""SentencePiece adapter. Exact Llama compatibility requires its original model file."""


class Llama2Tokenizer:
    def __init__(self, model_file=None):
        self.processor = None
        if model_file:
            import sentencepiece as spm

            self.processor = spm.SentencePieceProcessor(model_file=str(model_file))

    def _ready(self):
        if self.processor is None:
            raise ValueError("Load a SentencePiece model file first")
        return self.processor

    def train(self, text, vocabSize):
        import sentencepiece as spm
        import io

        if not text or vocabSize < 256:
            raise ValueError(
                "Provide nonempty training text and vocabulary size >= 256"
            )
        model = io.BytesIO()
        spm.SentencePieceTrainer.train(
            sentence_iterator=iter(text.splitlines()),
            model_writer=model,
            vocab_size=vocabSize,
            model_type="bpe",
            hard_vocab_limit=False,
            normalization_rule_name="identity",
            remove_extra_whitespaces=False,
        )
        self.processor = spm.SentencePieceProcessor(model_proto=model.getvalue())

    def encode(self, text):
        return self._ready().encode(text, out_type=int)

    def decode(self, ids):
        return self._ready().decode(ids)

    encoder = encode
    decoder = decode
