import json
from collections import Counter

from conftest import ROOT
from rag_cskh.ingest import load_chunks, split_section


def test_knowledge_base_size():
    chunks = load_chunks(ROOT / "data" / "kb", 300)
    assert len({c.doc_id for c in chunks}) == 50
    assert 1000 <= len(chunks) <= 1500
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert max(len(c.text) for c in chunks) <= 300


def test_expired_policy_is_marked():
    chunks = load_chunks(ROOT / "data" / "kb", 300)
    old = [c for c in chunks if c.doc_id == "CS-BAOHANH-2023"]
    assert old and all(c.status == "expired" for c in old)
    assert "HẾT HIỆU LỰC" in old[0].search_text


def test_tickets_are_labelled_and_grounded():
    tickets = [json.loads(line) for line in (ROOT / "data" / "eval" / "tickets.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(tickets) == 150
    assert Counter(t["type"] for t in tickets) == {"normal": 80, "missing_info": 25, "no_evidence": 25, "conflict": 20}
    sections = {c.section_key for c in load_chunks(ROOT / "data" / "kb", 300)}
    for t in tickets:
        assert t["expected_decision"] == {"normal": "answer", "missing_info": "clarify"}.get(t["type"], "escalate")
        assert all(g in sections for g in t["gold_sections"]), t["id"]
        if t["type"] in ("normal", "conflict"):
            assert t["gold_sections"], t["id"]
        if t["type"] == "conflict":
            assert len(t["gold_sections"]) == 2


def test_split_section_respects_limit():
    body = "Câu một rất dài. " * 40 + "\n\n- mục a\n- mục b"
    parts = split_section(body, 120)
    assert all(len(p) <= 120 for p in parts)
    assert "mục b" in parts[-1]
