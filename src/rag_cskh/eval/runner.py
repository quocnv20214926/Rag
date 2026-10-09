"""Evaluation run: 150 tickets x 3 architectures -> LLM judge -> metrics -> report.

Results are appended to reports/runs/<arch>.jsonl, so a run can be stopped (Ctrl+C, quota) and resumed;
finished tickets are skipped. LLM calls are also cached on disk.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path

from ..config import settings
from ..prompts import judge_prompt
from .metrics import METRIC_GROUPS, LOWER_IS_BETTER, compute_metrics


def load_tickets(path: Path | None = None, limit: int | None = None, types: list[str] | None = None) -> list[dict]:
    with (path or settings.eval_file).open(encoding="utf-8") as f:
        tickets = [json.loads(line) for line in f if line.strip()]
    if types:
        tickets = [t for t in tickets if t["type"] in types]
    if limit:                                    # take the same number of tickets per type
        by_type: dict[str, list] = {}
        for t in tickets:
            by_type.setdefault(t["type"], []).append(t)
        per = max(1, limit // len(by_type))
        tickets = [t for group in by_type.values() for t in group[:per]]
    return tickets


def _read_jsonl(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            out[rec["id"]] = rec
    return out


def run_pipelines(pipelines: dict, tickets: list[dict], out_dir: Path, log=print) -> dict[str, dict[str, dict]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for arch, pipe in pipelines.items():
        path = out_dir / f"{arch}.jsonl"
        done = _read_jsonl(path)
        todo = [t for t in tickets if t["id"] not in done]
        log(f"[{arch}] {len(done)} done, {len(todo)} to run")
        with path.open("a", encoding="utf-8") as f:
            for i, t in enumerate(todo, 1):
                res = pipe.run(t["ticket"]).to_dict()
                res["id"] = t["id"]
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
                f.flush()
                done[t["id"]] = res
                if i % 10 == 0 or i == len(todo):
                    log(f"  [{arch}] {i}/{len(todo)}  last={t['id']} decision={res['decision']} {res['latency_s']}s"
                        + (f" ERROR {res['error'][:80]}" if res["error"] else ""))
        results[arch] = {t["id"]: done[t["id"]] for t in tickets if t["id"] in done}
    return results


def judge_results(results: dict[str, dict[str, dict]], tickets: list[dict], judge_llm, kb, out_dir: Path, log=print) -> None:
    """LLM judge for answered tickets (accuracy, faithfulness, citation support). Cached."""
    by_id = {t["id"]: t for t in tickets}
    for arch, res in results.items():
        path = out_dir / f"judge_{arch}.jsonl"
        cache = _read_jsonl(path)
        n_new = 0
        with path.open("a", encoding="utf-8") as f:
            for tid, r in res.items():
                if r["decision"] != "answer":
                    continue
                t = by_id[tid]
                ids = r["citations"] or r["context_ids"]
                evidence = "\n\n".join(f"[{i}] {kb.by_id[i].text}" for i in ids if i in kb.by_id)
                key = hashlib.sha1(json.dumps([r["reply"], ids, t["reference"]], ensure_ascii=False).encode()).hexdigest()
                if tid in cache and cache[tid].get("key") == key:
                    r["judge"] = cache[tid]["judge"]
                    continue
                system, user, schema = judge_prompt(t["ticket"], t["reference"], t["key_facts"], r["reply"], evidence)
                j = judge_llm.json("judge", system, user, schema,
                                   payload={"answer": r["reply"], "key_facts": t["key_facts"], "evidence": evidence})
                r["judge"] = j
                f.write(json.dumps({"id": tid, "key": key, "judge": j}, ensure_ascii=False) + "\n")
                n_new += 1
        log(f"[judge {arch}] {n_new} new judgements")


def section_sizes(kb) -> dict[str, int]:
    return Counter(c.section_key for c in kb.chunks)


def evaluate(results, tickets, kb) -> dict[str, dict]:
    sizes = section_sizes(kb)
    metrics = {}
    for arch, res in results.items():
        rows = [{**t, **res[t["id"]]} for t in tickets if t["id"] in res]
        metrics[arch] = compute_metrics(rows, sizes)
    return metrics


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 0.01 and v != 0 else f"{v:.3f}"
    return str(v)


def write_report(metrics: dict[str, dict], results, tickets, out_dir: Path, meta: dict) -> Path:
    archs = list(metrics)
    lines = ["# Evaluation: Vector RAG vs Hybrid RAG vs Corrective RAG", "",
             f"- Ticket: {meta['n_tickets']} ({meta['types']})",
             f"- LLM: `{meta['llm']}` · Judge: `{meta['judge']}` · Embedding: `{meta['embedder']}` · Reranker: `{meta['reranker']}`",
             f"- Run at: {time.strftime('%Y-%m-%d %H:%M')}", ""]
    for group, keys in METRIC_GROUPS.items():
        lines += [f"## {group}", "", "| Metric | " + " | ".join(archs) + " |", "|---|" + "---|" * len(archs)]
        for k in keys:
            vals = [metrics[a].get(k) for a in archs]
            nums = [v for v in vals if isinstance(v, (int, float)) and v == v]
            best = (min(nums) if k in LOWER_IS_BETTER else max(nums)) if nums else None
            cells = [f"**{_fmt(v)}**" if best is not None and v == best and len(nums) > 1 else _fmt(v) for v in vals]
            lines.append(f"| {k} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## Decision confusion matrix (rows = expected, columns = predicted)", ""]
    for a in archs:
        lines += [f"**{a}**", "", "| expected \\ predicted | answer | clarify | escalate |", "|---|---|---|---|"]
        for exp in ("answer", "clarify", "escalate"):
            row = [r for t in tickets if t["expected_decision"] == exp and (r := results[a].get(t["id"]))]
            c = Counter(r["decision"] for r in row)
            lines.append(f"| {exp} | {c['answer']} | {c['clarify']} | {c['escalate']} |")
        lines.append("")
    path = out_dir / "summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps({"meta": meta, "metrics": metrics}, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(results, tickets, out_dir / "per_ticket.csv")
    _plot(metrics, out_dir)
    return path


def _write_csv(results, tickets, path: Path) -> None:
    import csv
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "type", "expected", "arch", "decision", "correct_decision", "judge_correct", "faithfulness",
                    "confidence", "rounds", "rewrites", "citations", "latency_s", "calls", "tokens", "ticket", "reply"])
        for t in tickets:
            for arch, res in results.items():
                r = res.get(t["id"])
                if not r:
                    continue
                j = r.get("judge") or {}
                w.writerow([t["id"], t["type"], t["expected_decision"], arch, r["decision"], r["decision"] == t["expected_decision"],
                            j.get("correct", ""), j.get("faithfulness", ""), r["confidence"], r["rounds"], " | ".join(r["rewrites"]),
                            " ".join(r["citations"]), r["latency_s"], r["usage"].get("calls"),
                            r["usage"].get("input_tokens", 0) + r["usage"].get("output_tokens", 0), t["ticket"], r["reply"]])


def _plot(metrics: dict[str, dict], out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    archs = list(metrics)
    panels = [("Retrieval", ["recall@5", "recall@10", "mrr", "ndcg@10", "context_precision"]),
              ("Answer & safety", ["decision_accuracy", "answer_accuracy", "faithfulness", "abstention_rate", "hallucination_rate"]),
              ("Per ticket type", ["decision_accuracy[normal]", "decision_accuracy[missing_info]", "decision_accuracy[no_evidence]",
                                    "decision_accuracy[conflict]"])]
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    width = 0.8 / len(archs)
    for ax, (title, keys) in zip(axes, panels):
        for i, a in enumerate(archs):
            vals = [metrics[a].get(k) or 0 for k in keys]
            vals = [0 if v != v else v for v in vals]
            ax.bar([x + i * width for x in range(len(keys))], vals, width, label=a)
        ax.set_xticks([x + width * (len(archs) - 1) / 2 for x in range(len(keys))])
        ax.set_xticklabels([k.replace("decision_accuracy", "acc") for k in keys], rotation=25, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "comparison.png", dpi=130)
    plt.close(fig)
