from .base import Pipeline
from .corrective import CorrectiveRAG
from .simple import HybridRAG, VectorRAG

ARCHITECTURES = {"vector_rag": VectorRAG, "hybrid_rag": HybridRAG, "corrective_rag": CorrectiveRAG}


def build_pipelines(kb, llm, reranker, names=None) -> dict[str, Pipeline]:
    names = names or list(ARCHITECTURES)
    return {n: ARCHITECTURES[n](kb, llm, reranker) for n in names}
