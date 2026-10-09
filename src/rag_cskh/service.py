"""Shared setup for the CLI, API and UI: knowledge base, LLM, reranker and the three pipelines."""

from __future__ import annotations

from functools import lru_cache

from .config import settings
from .llm import get_llm
from .pipelines import build_pipelines
from .rerank import get_reranker
from .retrieval import KnowledgeBase


@lru_cache(maxsize=1)
def get_service():
    kb = KnowledgeBase.load()
    llm = get_llm()
    reranker = get_reranker(llm=llm)
    return {"kb": kb, "llm": llm, "reranker": reranker, "pipelines": build_pipelines(kb, llm, reranker)}


def describe() -> dict:
    return {"llm": f"{settings.llm_provider}:{settings.llm_model}", "judge": f"{settings.llm_provider}:{settings.judge_model}",
            "embedder": f"{settings.embed_provider}:{settings.embed_model}", "reranker": settings.reranker}
