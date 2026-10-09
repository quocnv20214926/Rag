"""Embeddings: OpenAI (default) or HashEmbedder (offline, tests only).

Vectors are L2-normalised, so the dot product is the cosine similarity. Document embeddings are cached on
disk by content, so rebuilding the index does not call the API again for unchanged chunks.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .config import settings
from .text import bm25_tokens, strip_accents


def _normalize(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    return m / np.maximum(np.linalg.norm(m, axis=-1, keepdims=True), 1e-12)


class HashEmbedder:
    """Feature hashing of tokens + accent-stripped character trigrams. No real semantics - offline tests only."""

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.name = f"hash-{dim}"

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        plain = strip_accents(text.lower())
        feats = bm25_tokens(text) + [plain[i:i + 3] for i in range(len(plain) - 2)]
        for f in feats:
            h = int(hashlib.md5(f.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 64) & 1 else -1.0
        return v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return _normalize(np.stack([self._vec(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalize(self._vec(text)[None])[0]


class _CachedEmbedder:
    """Document embeddings cached on disk by content; query embeddings cached in memory."""

    def _init_cache(self, cache_dir: Path | None, tag: str):
        self.cache_path = (cache_dir or settings.cache_dir) / f"emb_{tag}.jsonl"
        self.cache = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                self.cache[rec["k"]] = rec["v"]
        self._query_cache: dict[str, np.ndarray] = {}

    def _embed(self, texts: list[str], kind: str) -> list[list[float]]:
        raise NotImplementedError

    def embed_documents(self, texts: list[str], batch: int = 100) -> np.ndarray:
        keys = [hashlib.sha1(t.encode()).hexdigest() for t in texts]
        todo = list({k: t for k, t in zip(keys, texts) if k not in self.cache}.items())
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("a", encoding="utf-8") as f:
            for i in range(0, len(todo), batch):
                part = todo[i:i + batch]
                vecs = self._embed([t for _, t in part], "document")
                for (k, _), v in zip(part, vecs):
                    self.cache[k] = v
                    f.write(json.dumps({"k": k, "v": v}) + "\n")
                print(f"  embedded {min(i + batch, len(todo))}/{len(todo)}", flush=True)
        return _normalize(np.array([self.cache[k] for k in keys]))

    def embed_query(self, text: str) -> np.ndarray:
        if text not in self._query_cache:
            self._query_cache[text] = _normalize(np.array(self._embed([text], "query")))[0]
        return self._query_cache[text]


class OpenAIEmbedder(_CachedEmbedder):
    """OpenAI text-embedding-3-small (multilingual)."""

    def __init__(self, model: str | None = None, dim: int | None = None, cache_dir: Path | None = None):
        from openai import OpenAI
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set (.env)")
        self.client = OpenAI(api_key=settings.openai_api_key, max_retries=5, timeout=60)
        self.model = model or settings.embed_model
        self.dim = dim or settings.embed_dim
        self.name = f"openai:{self.model}:{self.dim}"
        self._init_cache(cache_dir, f"{self.model}_{self.dim}")

    def _embed(self, texts, kind):
        resp = self.client.embeddings.create(model=self.model, input=texts, dimensions=self.dim)
        return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]


def get_embedder(provider: str | None = None):
    provider = provider or settings.embed_provider
    if provider == "openai":
        return OpenAIEmbedder()
    if provider == "hash":
        return HashEmbedder()
    raise ValueError(f"unknown EMBED_PROVIDER {provider}")
