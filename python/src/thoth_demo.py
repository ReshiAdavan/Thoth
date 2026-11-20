"""Loopback-only demo adapter for the actual Python and native implementations."""

import base64
import json
import uuid
from functools import lru_cache
from pathlib import Path
from thoth import load
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from basicTokenizer import BasicTokenizer
from regexTokenizer import RegexTokenizer
from gpt4Tokenizer import GPT4Tokenizer
from llama2Tokenizer import Llama2Tokenizer

MODELS = OrderedDict()
NAMES = {
    "local-trained",
    "local-basic",
    "local-regex",
    "local-rust",
    "local-gpt",
    "local-llama",
}


@lru_cache(maxsize=4)
def trained_model(name="local-regex"):
    root = Path(__file__).resolve().parents[2] / "models"
    if name == "local-llama":
        return Llama2Tokenizer(root / "thoth-sentencepiece-en-code-v1.model")
    if name == "local-rust":
        from thoth_native import NativeTokenizer

        return NativeTokenizer.from_json(
            (root / "thoth-basic-en-code-v1.json").read_text()
        )
    return load(
        root
        / (
            "thoth-basic-en-code-v1.json"
            if name == "local-basic"
            else "thoth-code-en-v1.json"
        )
    )


def create(name, trained=True):
    if trained and name in ("local-basic", "local-regex", "local-rust", "local-llama"):
        return trained_model(name)
    if name == "local-trained":
        return trained_model()
    if name == "local-basic":
        return BasicTokenizer()
    if name == "local-regex":
        return RegexTokenizer()
    if name == "local-gpt":
        return GPT4Tokenizer()
    if name == "local-llama":
        return Llama2Tokenizer()
    if name == "local-rust":
        from thoth_native import NativeTokenizer

        return NativeTokenizer()
    raise ValueError("Unknown local implementation")


def execute(data):
    name = data.get("name")
    if name not in NAMES:
        raise ValueError("Unknown local implementation")
    text = data.get("text", "")
    if not isinstance(text, str) or len(text.encode()) > 20000:
        raise ValueError("Use at most 20,000 UTF-8 bytes")
    operation = data.get("operation", "encode")
    if operation in ("train", "import"):
        t = create(name, trained=False)
        if operation == "import":
            if name != "local-llama":
                raise ValueError("Model-file import is for SentencePiece")
            import sentencepiece as spm

            raw = base64.b64decode(data.get("model", ""), validate=True)
            if len(raw) > 1_000_000:
                raise ValueError("Model exceeds 1 MB")
            t.processor = spm.SentencePieceProcessor(model_proto=raw)
        else:
            size = data.get("size", 300)
            if type(size) is not int or not 256 <= size <= 1024:
                raise ValueError("Vocabulary must be 256–1024")
            if name in ("local-gpt", "local-trained"):
                raise ValueError("Pretrained GPT adapter cannot be trained")
            t.train(text, size)
        key = str(uuid.uuid4())
        MODELS[key] = (name, t)
        while len(MODELS) > 16:
            MODELS.popitem(last=False)
        return {"modelId": key}
    if operation != "encode":
        raise ValueError("Unknown operation")
    key = data.get("modelId")
    if key:
        if key not in MODELS:
            raise ValueError("Demo model expired; train or import it again")
        stored_name, t = MODELS[key]
        if stored_name != name:
            raise ValueError("Model belongs to a different implementation")
    else:
        t = create(name)
    ids = t.encode(text)
    counts = (
        {name: len(ids), "cl100k_base": len(GPT4Tokenizer().encode(text))}
        if data.get("compare")
        else {}
    )
    if name == "local-llama":
        return {
            "ids": ids,
            "pieces": [t.processor.id_to_piece(i) for i in ids],
            "decoded": t.decode(ids),
            "counts": counts,
        }
    if name == "local-rust":
        model = json.loads(t.to_json())
        vocab = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(model["merges"], 256):
            vocab[i] = vocab[a] + vocab[b]
        byte_lists = [list(vocab[i]) for i in ids]
    else:
        byte_lists = [list(t.decode_bytes([i])) for i in ids]
    return {"ids": ids, "bytes": byte_lists, "counts": counts}


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        # Browser requests must arrive through the local Vite proxy; no CORS access.
        origin = self.headers.get("Origin", "")
        from urllib.parse import urlparse

        if origin and urlparse(origin).hostname not in ("127.0.0.1", "localhost"):
            self.send_error(403)
            return
        if self.path != "/api/tokenize":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 1_500_000:
                raise ValueError("Invalid request size")
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError("Expected JSON object")
            result = execute(data)
            status = 200
        except Exception as error:
            result = {"error": str(error)}
            status = 400
        payload = json.dumps(result).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass  # Client canceled a request; computation already completed.


def main():
    print("Thoth local implementations: http://127.0.0.1:8765", flush=True)
    ThreadingHTTPServer(("127.0.0.1", 8765), Handler).serve_forever()


if __name__ == "__main__":
    main()
