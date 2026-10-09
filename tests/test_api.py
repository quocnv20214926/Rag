from urllib.parse import quote

from fastapi.testclient import TestClient


def test_api_process_and_compare(kb, monkeypatch):
    from rag_cskh import service
    from rag_cskh.llm import FakeLLM
    from rag_cskh.pipelines import build_pipelines
    from rag_cskh.rerank import NoReranker
    svc = {"kb": kb, "llm": FakeLLM(), "reranker": NoReranker(), "pipelines": build_pipelines(kb, FakeLLM(), NoReranker())}
    import api.main as main
    monkeypatch.setattr(main, "get_service", lambda: svc)
    monkeypatch.setattr(service, "get_service", lambda: svc)
    client = TestClient(main.app)
    assert client.get("/health").json()["chunks"] == len(kb.chunks)
    r = client.post("/tickets/process", json={"ticket": "Máy giặt Vinto báo lỗi E21 là bị gì?", "arch": "hybrid_rag"}).json()
    assert r["decision"] in ("answer", "clarify", "escalate") and "sources" in r
    both = client.post("/tickets/compare", json={"ticket": "Cửa hàng mở cửa đến mấy giờ?"}).json()
    assert set(both) == {"vector_rag", "hybrid_rag", "corrective_rag"}
    cid = r["sources"][0]["chunk_id"]
    assert client.get(f"/chunks/{quote(cid, safe='')}").json()["chunk_id"] == cid
    assert client.post("/tickets/process", json={"ticket": "x", "arch": "bad"}).status_code == 422
