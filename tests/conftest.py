import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
# tests run offline: fake LLM, hash embeddings, no reranker
os.environ.update(LLM_PROVIDER="fake", EMBED_PROVIDER="hash", RERANKER="none")

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def kb(tmp_path_factory):
    from rag_cskh.embeddings import HashEmbedder
    from rag_cskh.retrieval import KnowledgeBase
    return KnowledgeBase.build(index_dir=tmp_path_factory.mktemp("index"), embedder=HashEmbedder())
