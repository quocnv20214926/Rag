"""Baselines.

Vector RAG : ticket -> vector search (top 20) -> top 6 chunks -> LLM answer.                 (1 LLM call)
Hybrid RAG : ticket -> BM25 + vector, RRF (top 20) -> reranker -> top 6 chunks -> LLM answer. (1 LLM call)

Both share the answer prompt with Corrective RAG (the LLM may also choose clarify/escalate), so the
differences between the three architectures come only from retrieval and the correction loop.
"""

from __future__ import annotations

from ..config import settings
from .base import Pipeline


class VectorRAG(Pipeline):
    name = "vector_rag"

    def _run(self, ticket, result, usage):
        hits = self.kb.search_vector(ticket, settings.top_k_retrieve)
        result.retrieved = [h.chunk.chunk_id for h in hits]
        context = hits[:settings.top_k_context]
        result.context_ids = [h.chunk.chunk_id for h in context]
        result.trace.append({"step": "retrieve", "method": "vector", "top": result.retrieved[:settings.top_k_context]})
        self.generate(ticket, context, result, usage)


class HybridRAG(Pipeline):
    name = "hybrid_rag"

    def _run(self, ticket, result, usage):
        hits = self.kb.search_hybrid(ticket, settings.top_k_retrieve)
        ranked = self.reranker.rerank(ticket, hits)
        result.retrieved = [h.chunk.chunk_id for h in ranked]
        context = ranked[:settings.top_k_context]
        result.context_ids = [h.chunk.chunk_id for h in context]
        result.trace.append({"step": "retrieve", "method": "bm25+vector (RRF) + rerank", "top": result.context_ids})
        self.generate(ticket, context, result, usage)
