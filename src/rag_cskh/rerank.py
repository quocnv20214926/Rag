"""Rerankers: reorder candidate chunks by relevance to the ticket.

  bge   cross-encoder BAAI/bge-reranker-v2-m3 (multilingual, runs locally, uses a GPU if available)
  llm   the LLM scores every chunk 0-10 in a single call
  none  keep the input order (tests)
"""

from __future__ import annotations

import math

from .config import settings
from .retrieval import Hit


class NoReranker:
    name = "none"

    def rerank(self, query: str, hits: list[Hit], top_n: int | None = None) -> list[Hit]:
        return hits[:top_n] if top_n else hits


class BGEReranker:
    name = "bge"

    def __init__(self, model: str | None = None):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model or settings.reranker_model, max_length=512)

    def rerank(self, query: str, hits: list[Hit], top_n: int | None = None) -> list[Hit]:
        if not hits:
            return hits
        scores = self.model.predict([(query, h.chunk.search_text) for h in hits], batch_size=16, show_progress_bar=False)
        out = [Hit(h.chunk, 1 / (1 + math.exp(-float(s))), "rerank") for h, s in zip(hits, scores)]
        out.sort(key=lambda h: -h.score)
        return out[:top_n] if top_n else out


class LLMReranker:
    name = "llm"

    def __init__(self, llm):
        self.llm = llm

    def rerank(self, query: str, hits: list[Hit], top_n: int | None = None, usage=None) -> list[Hit]:
        from .prompts import rerank_prompt
        if not hits:
            return hits
        system, user, schema = rerank_prompt(query, hits)
        data = self.llm.json("rerank", system, user, schema, payload={"query": query, "hits": hits}, usage=usage)
        score = {s["id"]: float(s["score"]) / 10 for s in data.get("scores", [])}
        out = [Hit(h.chunk, score.get(h.chunk.chunk_id, 0.0), "rerank") for h in hits]
        out.sort(key=lambda h: -h.score)
        return out[:top_n] if top_n else out


def get_reranker(kind: str | None = None, llm=None):
    kind = kind or settings.reranker
    if kind == "bge":
        return BGEReranker()
    if kind == "llm":
        return LLMReranker(llm)
    if kind == "none":
        return NoReranker()
    raise ValueError(f"unknown RERANKER {kind}")
