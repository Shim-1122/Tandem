"""CLIP text embeddings for the object detector's vocabulary (only needed to change the list of objects).

The detector (YOLO-World) recognizes an object by comparing image regions with the CLIP ViT-B/32 text embedding
of its name. This module tokenizes text the way CLIP does (byte-level BPE, lower case) and runs the CLIP text
encoder exported to ONNX, downloaded once from Hugging Face (Xenova/clip-vit-base-patch32, MIT license).
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

HF = "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/"
FILES = {"text_model.onnx": HF + "onnx/text_model.onnx", "clip_tokenizer.json": HF + "tokenizer.json"}
SOT, EOT = "<|startoftext|>", "<|endoftext|>"
_WORDS = re.compile(r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[^\W\d_]+|\d|[^\s\w]+|_+", re.IGNORECASE)


@lru_cache()
def bytes_to_unicode() -> dict:
    """CLIP's (and GPT-2's) reversible map from bytes to printable characters."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


class ClipTokenizer:
    def __init__(self, tokenizer_json):
        model = json.loads(Path(tokenizer_json).read_text(encoding="utf-8"))["model"]
        self.vocab = model["vocab"]
        merges = [tuple(m.split(" ")) if isinstance(m, str) else tuple(m) for m in model["merges"]]
        self.ranks = {m: i for i, m in enumerate(merges)}
        self.byte_map = bytes_to_unicode()
        self.cache = {}

    def bpe(self, token: str) -> list:
        if token in self.cache:
            return self.cache[token]
        word = list(token[:-1]) + [token[-1] + "</w>"]
        while len(word) > 1:
            pairs = [(self.ranks.get((a, b), 1 << 30), i) for i, (a, b) in enumerate(zip(word, word[1:]))]
            rank, i = min(pairs)
            if rank == 1 << 30:
                break
            first, second = word[i], word[i + 1]
            out, j = [], 0
            while j < len(word):                         # merge every occurrence of this pair
                if j < len(word) - 1 and word[j] == first and word[j + 1] == second:
                    out.append(first + second)
                    j += 2
                else:
                    out.append(word[j])
                    j += 1
            word = out
        self.cache[token] = word
        return word

    def encode(self, text: str) -> list:
        text = " ".join(text.split()).strip().lower()
        ids = [self.vocab[SOT]]
        for w in _WORDS.findall(text):
            mapped = "".join(self.byte_map[b] for b in w.encode("utf-8"))
            ids += [self.vocab[t] for t in self.bpe(mapped)]
        return ids[:76] + [self.vocab[EOT]]


class ClipTextEncoder:
    def __init__(self, model_dir="models"):
        from .face import download
        d = Path(model_dir)
        for name, url in FILES.items():
            download(url, d / name)
        import onnxruntime as ort
        self.tok = ClipTokenizer(d / "clip_tokenizer.json")
        self.session = ort.InferenceSession(str(d / "text_model.onnx"), providers=["CPUExecutionProvider"])

    def __call__(self, texts) -> np.ndarray:
        """(N, 512) unit-length embeddings, one text at a time (no padding)."""
        out = []
        for t in texts:
            ids = np.array([self.tok.encode(t)], np.int64)
            e = self.session.run(None, {"input_ids": ids})[0][0]
            out.append(e / np.linalg.norm(e))
        return np.array(out, np.float32)
