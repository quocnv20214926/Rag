from rag_cskh.retrieval import rrf
from rag_cskh.text import bm25_tokens, syllables


def test_vietnamese_tokens():
    toks = bm25_tokens("Bảo hành tủ lạnh 15.000.000 đồng")
    assert "bảo" in toks and "bảo_hành" in toks and "~bao" in toks
    assert "15000000" in syllables("15.000.000")


def test_bm25_handles_text_without_diacritics(kb):
    hits = kb.search_bm25("may giat bao loi e21", 5)
    assert hits and hits[0].chunk.doc_id == "HD-LOI-MAYGIAT"


def test_bm25_finds_policy(kb):
    hits = kb.search_bm25("phí đổi trả đổi ý", 5)
    assert any(h.chunk.section_key == "CS-DOITRA#doi-y" for h in hits)


def test_rrf_rewards_agreement():
    fused = rrf([["a", "b", "c"], ["b", "a", "d"]], k=60)
    assert fused["a"] == fused["b"] > fused["c"] > 0
    assert set(fused) == {"a", "b", "c", "d"}


def test_hybrid_and_multi_query(kb):
    hits = kb.search_many(["thời gian xử lý bảo hành", "bảo hành tại nhà"], 10)
    keys = {h.chunk.section_key for h in hits}
    assert "CS-BAOHANH-2025#thoi-gian-xu-ly" in keys and "CS-BAOHANH-2025#tai-nha" in keys
