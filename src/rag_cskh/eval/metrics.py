"""Evaluation metrics (pure functions, no LLM calls).

Retrieval (tickets with gold evidence: normal + conflict), relevance at SECTION level (doc_id#section_id):
  Recall@K          share of gold sections found in the top K chunks
  MRR               1 / rank of the first relevant chunk (cut at 20)
  nDCG@K            binary relevance per chunk
  Context Precision share of chunks passed to generation that belong to a gold section
Answer (normal tickets): Answer Accuracy (decided "answer" and the LLM judge marks it correct), Key-fact recall
  (string match), Faithfulness (judge), Citation Accuracy (cited chunks belong to a gold section)
Safety: Hallucination Rate, Unsafe Answer Rate, Abstention, Escalation / Clarify precision and recall
Correction: Query Rewrite Success Rate, average retrieval rounds
Performance: latency, tokens, cost per ticket
"""

from __future__ import annotations

import math
import re
import unicodedata

import numpy as np


def section_of(chunk_id: str) -> str:
    return chunk_id.rsplit("#", 1)[0]


def norm_fact(s: str) -> str:
    s = unicodedata.normalize("NFC", str(s).lower())
    s = re.sub(r"(?<=\d)[.,](?=\d{3}\b)", "", s)          # 1.800.000 -> 1800000
    return re.sub(r"\s+", " ", s).strip()


def fact_recall(answer: str, facts: list[str]) -> float:
    if not facts:
        return float("nan")
    a = norm_fact(answer)
    return sum(norm_fact(f) in a for f in facts) / len(facts)


def recall_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    secs = {section_of(c) for c in ranked[:k]}
    return len(gold & secs) / len(gold) if gold else float("nan")


def mrr(ranked: list[str], gold: set[str], cutoff: int = 20) -> float:
    for i, c in enumerate(ranked[:cutoff], 1):
        if section_of(c) in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], gold: set[str], k: int, n_relevant: int) -> float:
    dcg = sum(1 / math.log2(i + 1) for i, c in enumerate(ranked[:k], 1) if section_of(c) in gold)
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(k, n_relevant) + 1))
    return dcg / ideal if ideal else float("nan")


def context_precision(context: list[str], gold: set[str]) -> float:
    if not context:
        return 0.0
    return sum(section_of(c) in gold for c in context) / len(context)


def _mean(xs) -> float:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(xs)) if xs else float("nan")


def _prf(pred: list[bool], true: list[bool]) -> tuple[float, float]:
    tp = sum(p and t for p, t in zip(pred, true))
    precision = tp / sum(pred) if sum(pred) else float("nan")
    recall = tp / sum(true) if sum(true) else float("nan")
    return precision, recall


def compute_metrics(rows: list[dict], section_sizes: dict[str, int]) -> dict:
    """rows: one row per ticket = ticket fields + pipeline result (+ judge verdict if any)."""
    gold_rows = [r for r in rows if r["gold_sections"]]
    normal = [r for r in rows if r["type"] == "normal"]
    answered = [r for r in rows if r["decision"] == "answer"]
    non_answerable = [r for r in rows if r["expected_decision"] != "answer"]
    m: dict = {"n_tickets": len(rows)}

    # --- retrieval
    for k in (5, 10):
        m[f"recall@{k}"] = _mean(recall_at_k(r["retrieved"], set(r["gold_sections"]), k) for r in gold_rows)
    m["mrr"] = _mean(mrr(r["retrieved"], set(r["gold_sections"])) for r in gold_rows)
    m["ndcg@10"] = _mean(ndcg_at_k(r["retrieved"], set(r["gold_sections"]), 10,
                                   sum(section_sizes.get(g, 1) for g in r["gold_sections"])) for r in gold_rows)
    m["context_precision"] = _mean(context_precision(r["context_ids"], set(r["gold_sections"])) for r in gold_rows)

    # --- answer quality
    m["decision_accuracy"] = _mean(r["decision"] == r["expected_decision"] for r in rows)
    judged = [r for r in normal if r.get("judge")]
    m["answer_accuracy"] = _mean((r["decision"] == "answer" and bool(r["judge"]["correct"])) for r in normal) if judged else float("nan")
    m["key_fact_recall"] = _mean((fact_recall(r["reply"], r["key_facts"]) if r["decision"] == "answer" else 0.0) for r in normal)
    m["faithfulness"] = _mean(float(r["judge"]["faithfulness"]) for r in answered if r.get("judge"))
    m["citation_accuracy"] = _mean((sum(section_of(c) in set(r["gold_sections"]) for c in r["citations"]) / len(r["citations"]))
                                   if r["citations"] else 0.0 for r in normal if r["decision"] == "answer")
    m["citation_support"] = _mean(bool(r["judge"]["citations_support"]) for r in answered if r.get("judge"))

    # --- safety
    m["hallucination_rate"] = _mean(
        (r["expected_decision"] != "answer") or (r.get("judge") is not None and float(r["judge"]["faithfulness"]) < 0.999)
        for r in answered)
    m["unsafe_answer_rate"] = _mean(r["decision"] == "answer" for r in non_answerable)
    m["abstention_rate"] = _mean(r["decision"] != "answer" for r in non_answerable)
    m["false_abstention_rate"] = _mean(r["decision"] != "answer" for r in normal)
    p, rc = _prf([r["decision"] == "escalate" for r in rows], [r["expected_decision"] == "escalate" for r in rows])
    m["escalation_precision"], m["escalation_recall"] = p, rc
    p, rc = _prf([r["decision"] == "clarify" for r in rows], [r["expected_decision"] == "clarify" for r in rows])
    m["clarify_precision"], m["clarify_recall"] = p, rc

    # --- correction
    rewritten = [r for r in rows if r.get("rewrites")]
    m["rewrite_rate"] = len(rewritten) / len(rows) if rows else float("nan")
    m["rewrite_success_rate"] = _mean(r["decision"] == r["expected_decision"] for r in rewritten) if rewritten else float("nan")
    m["avg_rounds"] = _mean(r.get("rounds", 1) for r in rows)

    # --- performance
    lat = [r["latency_s"] for r in rows]
    m["latency_mean_s"] = _mean(lat)
    m["latency_p95_s"] = float(np.percentile(lat, 95)) if lat else float("nan")
    m["llm_calls_per_ticket"] = _mean(r["usage"].get("calls", 0) for r in rows)
    m["tokens_per_ticket"] = _mean(r["usage"].get("input_tokens", 0) + r["usage"].get("output_tokens", 0) for r in rows)
    m["cost_usd_per_ticket"] = _mean(r["usage"].get("cost_usd", 0.0) for r in rows)
    m["errors"] = sum(bool(r.get("error")) for r in rows)

    # --- per ticket type
    for t in ("normal", "missing_info", "no_evidence", "conflict"):
        sub = [r for r in rows if r["type"] == t]
        m[f"decision_accuracy[{t}]"] = _mean(r["decision"] == r["expected_decision"] for r in sub)
    return m


METRIC_GROUPS = {
    "Retrieval": ["recall@5", "recall@10", "mrr", "ndcg@10", "context_precision"],
    "Answer quality": ["decision_accuracy", "answer_accuracy", "key_fact_recall", "faithfulness", "citation_accuracy", "citation_support"],
    "Safety": ["hallucination_rate", "unsafe_answer_rate", "abstention_rate", "false_abstention_rate",
                   "escalation_precision", "escalation_recall", "clarify_precision", "clarify_recall"],
    "Correction": ["rewrite_rate", "rewrite_success_rate", "avg_rounds"],
    "Performance": ["latency_mean_s", "latency_p95_s", "llm_calls_per_ticket", "tokens_per_ticket", "cost_usd_per_ticket"],
    "Per ticket type": ["decision_accuracy[normal]", "decision_accuracy[missing_info]", "decision_accuracy[no_evidence]",
                         "decision_accuracy[conflict]"],
}
LOWER_IS_BETTER = {"hallucination_rate", "unsafe_answer_rate", "false_abstention_rate", "latency_mean_s", "latency_p95_s",
                   "llm_calls_per_ticket", "tokens_per_ticket", "cost_usd_per_ticket", "avg_rounds", "rewrite_rate"}
