"""Index and retrieval: BM25, vector (cosine), hybrid (Reciprocal Rank Fusion).

    kb = KnowledgeBase.load()            # reads data/index (built by `python -m rag_cskh.cli index`)
    kb.search_bm25("đổi trả 7 ngày", k=20)
    kb.search_vector("đổi trả 7 ngày", k=20)
    kb.search_hybrid("đổi trả 7 ngày", k=20)
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import settings
from .embeddings import get_embedder
from .ingest import Chunk, load_chunks, read_chunks, save_chunks
from .text import bm25_tokens


@dataclass
class Hit:
    chunk: Chunk
    score: float
    source: str           # bm25 | vector | hybrid | rerank


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(docs)
        self.len = np.array([len(d) for d in docs], dtype=np.float32)
        self.avgdl = float(self.len.mean()) if self.n else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, d in enumerate(docs):
            for term, tf in Counter(d).items():
                self.postings[term].append((i, tf))
        self.idf = {t: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        s = np.zeros(self.n, dtype=np.float32)
        for term in dict.fromkeys(query):          # unique terms, original order
            if term not in self.postings:
                continue
            idf = self.idf[term]
            for i, tf in self.postings[term]:
                s[i] += idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.len[i] / self.avgdl))
        return s


def rrf(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion: score(d) = sum 1 / (k + rank)."""
    fused: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for r, cid in enumerate(ranking, 1):
            fused[cid] += 1.0 / (k + r)
    return fused


class KnowledgeBase:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, embedder):
        self.chunks = chunks
        self.by_id = {c.chunk_id: c for c in chunks}
        self.vectors = vectors
        self.embedder = embedder
        self.bm25 = BM25([bm25_tokens(c.search_text) for c in chunks])

    # ------------------------------------------------------------------ build / load
    @classmethod
    def build(cls, kb_dir: Path | None = None, index_dir: Path | None = None, embedder=None) -> "KnowledgeBase":
        index_dir = index_dir or settings.index_dir
        chunks = load_chunks(kb_dir)
        embedder = embedder or get_embedder()
        vectors = embedder.embed_documents([c.search_text for c in chunks])
        index_dir.mkdir(parents=True, exist_ok=True)
        save_chunks(chunks, index_dir / "chunks.jsonl")
        np.save(index_dir / "vectors.npy", vectors)
        (index_dir / "meta.json").write_text(json.dumps({"embedder": embedder.name, "dim": int(vectors.shape[1]),
                                                         "chunks": len(chunks)}, indent=2), encoding="utf-8")
        return cls(chunks, vectors, embedder)

    @classmethod
    def load(cls, index_dir: Path | None = None, embedder=None) -> "KnowledgeBase":
        index_dir = index_dir or settings.index_dir
        if not (index_dir / "chunks.jsonl").exists():
            raise FileNotFoundError(f"No index in {index_dir}. Run: python -m rag_cskh.cli index")
        meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
        embedder = embedder or get_embedder()
        if embedder.name != meta["embedder"]:
            raise ValueError(f"Index built with {meta['embedder']}, current embedder is {embedder.name}; rebuild the index")
        return cls(read_chunks(index_dir / "chunks.jsonl"), np.load(index_dir / "vectors.npy"), embedder)

    # ------------------------------------------------------------------ search
    def search_bm25(self, query: str, k: int = 20) -> list[Hit]:
        s = self.bm25.scores(bm25_tokens(query))
        idx = np.argsort(-s)[:k]
        return [Hit(self.chunks[i], float(s[i]), "bm25") for i in idx if s[i] > 0]

    def search_vector(self, query: str, k: int = 20) -> list[Hit]:
        q = self.embedder.embed_query(query)
        s = self.vectors @ q
        idx = np.argsort(-s)[:k]
        return [Hit(self.chunks[i], float(s[i]), "vector") for i in idx]

    def search_hybrid(self, query: str, k: int = 20, pool: int = 40) -> list[Hit]:
        b = [h.chunk.chunk_id for h in self.search_bm25(query, pool)]
        v = [h.chunk.chunk_id for h in self.search_vector(query, pool)]
        fused = rrf([b, v], k=settings.rrf_k)
        best = sorted(fused.items(), key=lambda x: -x[1])[:k]
        return [Hit(self.by_id[cid], score, "hybrid") for cid, score in best]

    def search_many(self, queries: list[str], k: int = 20) -> list[Hit]:
        """Hybrid search for several queries (original + rewritten), fused with RRF."""
        rankings = [[h.chunk.chunk_id for h in self.search_hybrid(q, k)] for q in queries if q.strip()]
        fused = rrf(rankings, k=settings.rrf_k)
        best = sorted(fused.items(), key=lambda x: -x[1])[:k]
        return [Hit(self.by_id[cid], score, "hybrid") for cid, score in best]
