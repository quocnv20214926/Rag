"""Corrective RAG for customer-support tickets.

    ticket ──> analysis (LLM: intent, product, missing info, search queries)
          ──> hybrid BM25 + vector retrieval (several queries, RRF) ──> reranking
          ──> evidence grading (LLM) ──┬── sufficient    ──> handling steps + reply with citations (accepted chunks only)
                                        ├── insufficient  ──> rewrite the query and retrieve again (up to CRAG_MAX_ROUNDS)
                                        ├── missing_info  ──> clarifying questions
                                        └── no_evidence / ──> no conclusion, escalate to a human agent
                                            conflict

Evidence accumulates across rounds (pool); every round reranks the whole pool against the original ticket.
Customer-facing texts below stay in Vietnamese.
"""

from __future__ import annotations

from ..config import settings
from ..prompts import analyze_prompt, grade_prompt
from .base import Pipeline

GRADE_TOP = 8           # chunks passed to the grading step
MAX_PER_DOC = 3         # max chunks per document in GRADE_TOP (keeps catalogs from crowding out policies)


def diversify(ranked, k: int = GRADE_TOP, per_doc: int = MAX_PER_DOC):
    """Top k chunks in rerank order with at most per_doc chunks per document; backfill from the rest if short."""
    out, count, rest = [], {}, []
    for h in ranked:
        if count.get(h.chunk.doc_id, 0) < per_doc:
            out.append(h)
            count[h.chunk.doc_id] = count.get(h.chunk.doc_id, 0) + 1
        else:
            rest.append(h)
        if len(out) == k:
            return out
    return out + rest[:k - len(out)]


class CorrectiveRAG(Pipeline):
    name = "corrective_rag"

    def __init__(self, kb, llm, reranker=None, max_rounds: int | None = None):
        super().__init__(kb, llm, reranker)
        self.max_rounds = max_rounds or settings.max_rounds

    def _run(self, ticket, result, usage):
        # 1. analyse the ticket
        system, user, schema = analyze_prompt(ticket)
        analysis = self.llm.json("analyze", system, user, schema, payload={"ticket": ticket}, usage=usage)
        result.trace.append({"step": "analyze", **analysis})

        queries = [ticket] + [q for q in analysis.get("search_queries", []) if q][:3]
        used: list[str] = []
        pool: dict[str, object] = {}
        grade, ranked = {}, []
        for round_no in range(1, self.max_rounds + 1):
            # 2. hybrid retrieval, then rerank the whole pool against the original ticket
            for h in self.kb.search_many(queries, settings.top_k_retrieve):
                pool.setdefault(h.chunk.chunk_id, h)
            used += queries
            ranked = self.reranker.rerank(ticket, list(pool.values()))
            top = diversify(ranked)
            result.trace.append({"step": "retrieve", "round": round_no, "queries": queries,
                                 "top": [h.chunk.chunk_id for h in top]})
            # 3. grade the evidence
            system, user, schema = grade_prompt(ticket, analysis, top, round_no, self.max_rounds, used)
            grade = self.llm.json("grade", system, user, schema,
                                  payload={"ticket": ticket, "analysis": analysis, "hits": top, "round": round_no,
                                           "max_rounds": self.max_rounds}, usage=usage)
            result.trace.append({"step": "grade", "round": round_no, **grade})
            result.rounds = round_no
            # analysis found nothing missing but the grader asks the customer anyway -> usually the right policy was not retrieved: search again
            if (grade.get("verdict") == "missing_info" and not analysis.get("missing_info")
                    and round_no < self.max_rounds):
                grade["verdict"] = "insufficient"
                if not (grade.get("rewritten_query") or "").strip():
                    grade["rewritten_query"] = f"quy định {analysis.get('intent', '')} {' '.join(analysis.get('entities', [])[:2])}".strip()
                result.trace[-1]["override"] = "missing_info -> insufficient"
            if grade.get("verdict") != "insufficient" or round_no == self.max_rounds:
                break
            # 4. rewrite the query
            new_q = (grade.get("rewritten_query") or "").strip()
            if not new_q or new_q in used:
                break
            result.rewrites.append(new_q)
            queries = [new_q]

        result.retrieved = [h.chunk.chunk_id for h in ranked[:settings.top_k_retrieve]]
        verdict = grade.get("verdict", "no_evidence")
        conf = float(max(0.0, min(1.0, grade.get("confidence", 0.0))))
        by_id = {h.chunk.chunk_id: h for h in ranked}

        if verdict == "conflict":
            # a conflict only counts when the conflicting chunks come from at least two different documents
            ids = [i for i in grade.get("conflict_ids", []) if i in by_id]
            if len({by_id[i].chunk.doc_id for i in ids}) < 2:
                verdict = "missing_info" if analysis.get("missing_info") else "sufficient"
                grade["relevant_ids"] = list(dict.fromkeys(grade.get("relevant_ids", []) + ids))
                result.trace.append({"step": "check", "override": f"conflict within one document -> {verdict}"})

        if verdict == "sufficient":
            accepted = [by_id[i] for i in grade.get("relevant_ids", []) if i in by_id and by_id[i].chunk.status == "active"]
            if not accepted:                                   # grader returned no valid id -> do not answer
                verdict = "no_evidence"
            else:
                result.context_ids = [h.chunk.chunk_id for h in accepted]
                self.generate(ticket, accepted, result, usage, forced=True)
                result.confidence = round(min(conf, result.confidence), 3)
                return

        if verdict == "missing_info":
            questions = grade.get("missing_info") or analysis.get("missing_info") or ["Anh/chị vui lòng cung cấp thêm thông tin."]
            result.decision = "clarify"
            result.clarifying_questions = questions
            result.reply = ("Dạ, để Lam Sơn hỗ trợ chính xác, anh/chị vui lòng cho biết thêm: "
                            + "; ".join(q.rstrip("?.") for q in questions) + ".")
            result.handling_steps = ["Hỏi lại khách các thông tin còn thiếu trước khi xử lý."]
            result.context_ids = [h.chunk.chunk_id for h in ranked[:3]]
        else:  # conflict / no_evidence / insufficient after the last round
            result.decision = "escalate"
            if verdict == "conflict":
                ids = [i for i in grade.get("conflict_ids", []) if i in by_id]
                result.citations = ids
                result.context_ids = ids
                result.reason = "Tài liệu mâu thuẫn: " + grade.get("explanation", "")
                result.handling_steps = ["Không kết luận với khách.", "Chuyển trưởng ca xác nhận chính sách áp dụng; ghi rõ các nguồn mâu thuẫn."]
            else:
                result.reason = "Không tìm thấy căn cứ trong tài liệu: " + grade.get("explanation", "")
                result.handling_steps = ["Không kết luận với khách.", "Chuyển nhân viên/bộ phận phụ trách xử lý."]
            result.reply = ("Dạ, Lam Sơn đã ghi nhận yêu cầu của anh/chị. Yêu cầu này cần bộ phận chuyên trách kiểm tra thêm; "
                            "nhân viên sẽ liên hệ lại anh/chị trong thời gian sớm nhất.")
        result.confidence = round(conf, 3)
