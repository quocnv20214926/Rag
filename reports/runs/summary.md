# Evaluation: Vector RAG vs Hybrid RAG vs Corrective RAG

- Ticket: 150 ({'normal': 80, 'missing_info': 25, 'no_evidence': 25, 'conflict': 20})
- LLM: `openai:gpt-4o-mini` · Judge: `openai:gpt-4.1-mini` · Embedding: `openai:text-embedding-3-small` · Reranker: `bge`
- Run at: 2026-10-10 03:45

## Retrieval

| Metric | vector_rag | hybrid_rag | corrective_rag |
|---|---|---|---|
| recall@5 | 0.765 | 0.865 | **0.885** |
| recall@10 | 0.825 | 0.910 | **0.930** |
| mrr | 0.734 | 0.862 | **0.871** |
| ndcg@10 | 0.663 | 0.754 | **0.771** |
| context_precision | 0.220 | 0.238 | **0.704** |

## Answer quality

| Metric | vector_rag | hybrid_rag | corrective_rag |
|---|---|---|---|
| decision_accuracy | 0.553 | 0.607 | **0.807** |
| answer_accuracy | 0.787 | **0.825** | 0.812 |
| key_fact_recall | 0.750 | **0.800** | 0.775 |
| faithfulness | 0.921 | 0.893 | **0.933** |
| citation_accuracy | 0.688 | 0.666 | **0.713** |
| citation_support | 0.947 | 0.942 | **0.987** |

## Safety

| Metric | vector_rag | hybrid_rag | corrective_rag |
|---|---|---|---|
| hallucination_rate | 0.337 | 0.375 | **0.167** |
| unsafe_answer_rate | 0.429 | 0.429 | **0.114** |
| abstention_rate | 0.571 | 0.571 | **0.886** |
| false_abstention_rate | 0.188 | **0.075** | 0.125 |
| escalation_precision | 0.667 | **1.000** | 0.805 |
| escalation_recall | 0.044 | 0.022 | **0.733** |
| clarify_precision | 0.308 | 0.356 | **0.581** |
| clarify_recall | 0.640 | 0.640 | **0.720** |

## Correction

| Metric | vector_rag | hybrid_rag | corrective_rag |
|---|---|---|---|
| rewrite_rate | **0.000** | **0.000** | 0.107 |
| rewrite_success_rate | — | — | 0.812 |
| avg_rounds | **1.000** | **1.000** | 1.113 |

## Performance

| Metric | vector_rag | hybrid_rag | corrective_rag |
|---|---|---|---|
| latency_mean_s | **2.733** | 8.701 | 21.339 |
| latency_p95_s | **3.557** | 9.376 | 52.714 |
| llm_calls_per_ticket | **1.000** | **1.000** | 2.633 |
| tokens_per_ticket | **1429.920** | 1447.193 | 4939.200 |
| cost_usd_per_ticket | **0.0003** | 0.0003 | 0.0009 |

## Per ticket type

| Metric | vector_rag | hybrid_rag | corrective_rag |
|---|---|---|---|
| decision_accuracy[normal] | 0.812 | **0.925** | 0.875 |
| decision_accuracy[missing_info] | 0.640 | 0.640 | **0.720** |
| decision_accuracy[no_evidence] | 0.080 | 0.000 | **0.680** |
| decision_accuracy[conflict] | 0.000 | 0.050 | **0.800** |

## Decision confusion matrix (rows = expected, columns = predicted)

**vector_rag**

| expected \ predicted | answer | clarify | escalate |
|---|---|---|---|
| answer | 65 | 14 | 1 |
| clarify | 9 | 16 | 0 |
| escalate | 21 | 22 | 2 |

**hybrid_rag**

| expected \ predicted | answer | clarify | escalate |
|---|---|---|---|
| answer | 74 | 6 | 0 |
| clarify | 9 | 16 | 0 |
| escalate | 21 | 23 | 1 |

**corrective_rag**

| expected \ predicted | answer | clarify | escalate |
|---|---|---|---|
| answer | 70 | 6 | 4 |
| clarify | 3 | 18 | 4 |
| escalate | 5 | 7 | 33 |
