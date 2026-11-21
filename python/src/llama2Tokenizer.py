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

        if not text or type(vocabSize) is not int or vocabSize < 259:
            raise ValueError(
                "Provide nonempty training text and vocabulary size >= 259 "
                "(256 bytes plus special tokens; observed characters need additional slots)"
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
            add_dummy_prefix=False,
            byte_fallback=True,
            character_coverage=1.0,
        )
        self.processor = spm.SentencePieceProcessor(model_proto=model.getvalue())

    def encode(self, text):
        processor = self._ready()
        ids = processor.encode(text, out_type=int)
        if processor.decode(ids) == text:
            return ids
        # SentencePiece reserves literal U+2581 as a whitespace marker. Even
        # identity normalization is not universally reversible. Use explicit
        # byte tokens for exceptional input, without changing the model format.
        fallback = []
        for byte in text.encode("utf-8"):
            piece = f"<0x{byte:02X}>"
            token = processor.piece_to_id(piece)
            if processor.id_to_piece(token) != piece or not processor.is_byte(token):
                raise ValueError(
                    "This SentencePiece model cannot preserve the input exactly"
                )
            fallback.append(token)
        if processor.decode(fallback) != text:
            raise ValueError(
                "This SentencePiece model cannot preserve the input exactly"
            )
        return fallback

    def decode(self, ids):
        processor = self._ready()
        ids = list(ids)
        if any(
            type(i) is not int or not 0 <= i < processor.get_piece_size() for i in ids
        ):
            raise ValueError("Token IDs must be integers within the model vocabulary")
        return processor.decode(ids)

    def encode_batch(self, texts):
        return [self.encode(text) for text in texts]

    def save(self, path):
        from pathlib import Path

        Path(path).write_bytes(self._ready().serialized_model_proto())

    encoder = encode
    decoder = decode
