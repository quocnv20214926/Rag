from rag_cskh.llm import FakeLLM, Usage
from rag_cskh.pipelines import build_pipelines
from rag_cskh.rerank import NoReranker


def test_all_architectures_return_valid_results(kb):
    pipes = build_pipelines(kb, FakeLLM(), NoReranker())
    for name, p in pipes.items():
        r = p.run("Máy giặt Vinto báo lỗi E21 là bị gì?")
        assert r.decision in ("answer", "clarify", "escalate")
        assert set(r.citations) <= set(r.context_ids) or r.decision != "answer"
        assert r.usage["calls"] >= 1 and r.latency_s >= 0
        assert not r.error, (name, r.error)


def test_corrective_rag_asks_for_missing_information(kb):
    p = build_pipelines(kb, FakeLLM(), NoReranker(), ["corrective_rag"])["corrective_rag"]
    r = p.run("Giá bao nhiêu vậy shop?")
    assert r.decision == "clarify" and r.clarifying_questions


def test_corrective_rag_rewrites_then_escalates_without_evidence(kb):
    p = build_pipelines(kb, FakeLLM(), NoReranker(), ["corrective_rag"])["corrective_rag"]
    r = p.run("Cho mình hỏi có patin skateboard flycam drone gopro kayak hoverboard không vậy")
    assert r.decision == "escalate"
    assert r.rounds >= 2 and r.rewrites
    assert sum(1 for s in r.trace if s["step"] == "grade") == r.rounds


def test_corrective_rag_only_cites_accepted_evidence(kb):
    p = build_pipelines(kb, FakeLLM(), NoReranker(), ["corrective_rag"])["corrective_rag"]
    r = p.run("Máy giặt Vinto báo lỗi E21 thoát nước là bị gì?")
    assert r.decision == "answer"
    accepted = next(s for s in reversed(r.trace) if s["step"] == "grade")["relevant_ids"]
    assert set(r.citations) <= set(accepted)


def test_usage_cost(monkeypatch):
    from rag_cskh.config import settings
    monkeypatch.setattr(settings, "price_in_per_m", 0.15)
    u = Usage()
    u.add("generate", 1_000_000, 0, 1.0, cached=False)
    assert u.calls == 1 and u.cost_usd > 0


def test_diversify_caps_chunks_per_document():
    from types import SimpleNamespace as NS
    from rag_cskh.pipelines.corrective import diversify
    hits = [NS(chunk=NS(doc_id="SP")) for _ in range(6)] + [NS(chunk=NS(doc_id=d)) for d in ("CS", "HD")]
    top = diversify(hits, k=5, per_doc=3)
    assert [h.chunk.doc_id for h in top] == ["SP", "SP", "SP", "CS", "HD"]
    assert len(diversify(hits[:4], k=5, per_doc=3)) == 4          # backfilled from the rest


def test_conflict_within_one_document_is_not_a_conflict(kb):
    from rag_cskh.llm import FakeLLM
    from rag_cskh.pipelines.corrective import CorrectiveRAG
    from rag_cskh.rerank import NoReranker

    class OneDocConflict(FakeLLM):
        def _analyze(self, p):
            return {"intent": "bảo hành", "product": None, "entities": [], "missing_info": [], "search_queries": []}

        def _grade(self, p):
            ids = [h.chunk.chunk_id for h in p["hits"] if h.chunk.doc_id == "CS-BAOHANH-2025"][:2]
            return {"claims": [], "verdict": "conflict", "relevant_ids": [], "conflict_ids": ids, "missing_info": [],
                    "rewritten_query": "", "explanation": "", "confidence": 0.8}

    res = CorrectiveRAG(kb, OneDocConflict(), NoReranker()).run("Thời gian xử lý bảo hành mất bao lâu, có được mượn máy không?")
    assert res.decision == "answer"
    assert any(s.get("step") == "check" for s in res.trace)
