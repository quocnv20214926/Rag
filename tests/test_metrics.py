import math

from rag_cskh.eval.metrics import compute_metrics, context_precision, fact_recall, mrr, ndcg_at_k, norm_fact, recall_at_k

R = ["A#x#0", "B#y#0", "A#x#1", "C#z#0"]


def test_retrieval_metrics():
    assert recall_at_k(R, {"A#x", "C#z"}, 2) == 0.5
    assert recall_at_k(R, {"A#x", "C#z"}, 4) == 1.0
    assert mrr(R, {"B#y"}) == 0.5
    assert mrr(R, {"Q#q"}) == 0.0
    assert math.isclose(ndcg_at_k(R, {"A#x"}, 10, 2), (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3)))
    assert context_precision(R[:2], {"A#x"}) == 0.5


def test_fact_matching_normalises_numbers():
    assert norm_fact("1.800.000 đồng") == "1800000 đồng"
    assert fact_recall("Phí là 1800000 đồng, trong 15 ngày làm việc", ["1.800.000", "15 ngày làm việc"]) == 1.0
    assert fact_recall("không rõ", ["15 ngày"]) == 0.0


def _row(tid, typ, exp, dec, **kw):
    base = dict(id=tid, type=typ, expected_decision=exp, decision=dec, gold_sections=kw.pop("gold", []), key_facts=kw.pop("facts", []),
                retrieved=kw.pop("retrieved", []), context_ids=[], citations=kw.pop("citations", []), reply=kw.pop("reply", ""),
                rewrites=kw.pop("rewrites", []), rounds=1, latency_s=1.0, usage={"calls": 1, "input_tokens": 10, "output_tokens": 5,
                                                                                    "cost_usd": 0.001}, judge=kw.pop("judge", None))
    return base


def test_safety_and_decision_metrics():
    rows = [_row("1", "normal", "answer", "answer", gold=["A#x"], facts=["7 ngày"], reply="trong 7 ngày", citations=["A#x#0"],
                 retrieved=["A#x#0"], judge={"correct": True, "faithfulness": 1.0, "citations_support": True}),
            _row("2", "no_evidence", "escalate", "answer", judge={"correct": False, "faithfulness": 0.0, "citations_support": False}),
            _row("3", "conflict", "escalate", "escalate", gold=["A#x", "B#y"], retrieved=["B#y#0"]),
            _row("4", "missing_info", "clarify", "clarify", rewrites=["q"])]
    m = compute_metrics(rows, {"A#x": 2, "B#y": 1})
    assert m["decision_accuracy"] == 0.75
    assert m["answer_accuracy"] == 1.0
    assert m["hallucination_rate"] == 0.5          # 1 of 2 answers is unsupported (no-evidence ticket)
    assert m["abstention_rate"] == 2 / 3
    assert m["escalation_precision"] == 1.0 and m["escalation_recall"] == 0.5
    assert m["rewrite_success_rate"] == 1.0
    assert m["citation_accuracy"] == 1.0
