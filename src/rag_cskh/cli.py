"""Command line.

    python -m rag_cskh.cli index                         # preprocess the KB and build the index (OpenAI embeddings)
    python -m rag_cskh.cli ask "Tủ lạnh lỗi bảo hành bao lâu?" --arch corrective_rag
    python -m rag_cskh.cli retrieval                     # retrieval-only evaluation (no answer generation)
    python -m rag_cskh.cli eval --limit 20               # quick run on 20 tickets (5 per type)
    python -m rag_cskh.cli eval                          # full run: 150 tickets x 3 architectures + judge + report
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

from .config import settings


def cmd_index(args):
    from .retrieval import KnowledgeBase
    kb = KnowledgeBase.build()
    docs = len({c.doc_id for c in kb.chunks})
    print(f"Indexed {docs} documents, {len(kb.chunks)} chunks -> {settings.index_dir}")


def cmd_ask(args):
    from .service import get_service
    svc = get_service()
    res = svc["pipelines"][args.arch].run(args.ticket)
    print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))


def cmd_retrieval(args):
    """Retrieval-only comparison: vector, BM25, hybrid, hybrid + rerank (no answer generation)."""
    from .eval.metrics import mrr, ndcg_at_k, recall_at_k
    from .eval.runner import load_tickets, section_sizes
    from .rerank import get_reranker
    from .retrieval import KnowledgeBase
    kb = KnowledgeBase.load()
    reranker = get_reranker()
    sizes = section_sizes(kb)
    tickets = [t for t in load_tickets() if t["gold_sections"]]
    methods = {"vector": lambda q: kb.search_vector(q, 20), "bm25": lambda q: kb.search_bm25(q, 20),
               "hybrid": lambda q: kb.search_hybrid(q, 20),
               f"hybrid+rerank({reranker.name})": lambda q: reranker.rerank(q, kb.search_hybrid(q, 20))}
    print(f"{len(tickets)} tickets with gold evidence")
    for name, fn in methods.items():
        rows = []
        for t in tickets:
            ranked = [h.chunk.chunk_id for h in fn(t["ticket"])]
            g = set(t["gold_sections"])
            rows.append((recall_at_k(ranked, g, 5), recall_at_k(ranked, g, 10), mrr(ranked, g),
                         ndcg_at_k(ranked, g, 10, sum(sizes[s] for s in g))))
        avg = [sum(r[i] for r in rows) / len(rows) for i in range(4)]
        print(f"{name:28s} Recall@5 {avg[0]:.3f}  Recall@10 {avg[1]:.3f}  MRR {avg[2]:.3f}  nDCG@10 {avg[3]:.3f}")


def cmd_eval(args):
    from .eval.runner import evaluate, judge_results, load_tickets, run_pipelines, write_report
    from .llm import get_llm
    from .service import describe, get_service
    svc = get_service()
    names = args.arch.split(",") if args.arch != "all" else list(svc["pipelines"])
    tickets = load_tickets(limit=args.limit, types=args.types.split(",") if args.types else None)
    out_dir = settings.reports_dir / ("runs" if not args.limit else f"runs_limit{args.limit}")
    results = run_pipelines({n: svc["pipelines"][n] for n in names}, tickets, out_dir)
    if not args.no_judge:
        judge = get_llm(model=settings.judge_model) if settings.llm_provider != "fake" else get_llm()
        judge_results(results, tickets, judge, svc["kb"], out_dir)
    metrics = evaluate(results, tickets, svc["kb"])
    meta = {**describe(), "n_tickets": len(tickets), "types": dict(Counter(t["type"] for t in tickets))}
    path = write_report(metrics, results, tickets, out_dir, meta)
    print(path.read_text(encoding="utf-8"))


def main(argv=None):
    p = argparse.ArgumentParser(prog="rag_cskh")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("index")
    a = sub.add_parser("ask")
    a.add_argument("ticket")
    a.add_argument("--arch", default="corrective_rag", choices=["vector_rag", "hybrid_rag", "corrective_rag"])
    sub.add_parser("retrieval")
    e = sub.add_parser("eval")
    e.add_argument("--arch", default="all")
    e.add_argument("--limit", type=int, default=None)
    e.add_argument("--types", default=None, help="normal,missing_info,no_evidence,conflict")
    e.add_argument("--no-judge", action="store_true")
    args = p.parse_args(argv)
    {"index": cmd_index, "ask": cmd_ask, "retrieval": cmd_retrieval, "eval": cmd_eval}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
