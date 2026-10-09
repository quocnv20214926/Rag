"""Shared result type and generation step for all three architectures."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

from ..llm import Usage
from ..prompts import answer_prompt


@dataclass
class TicketResult:
    arch: str
    ticket: str
    decision: str = "escalate"                 # answer | clarify | escalate
    reply: str = ""                            # drafted reply to the customer
    handling_steps: list = field(default_factory=list)
    clarifying_questions: list = field(default_factory=list)
    citations: list = field(default_factory=list)       # cited chunk ids
    context_ids: list = field(default_factory=list)     # chunks passed to the generation step
    retrieved: list = field(default_factory=list)       # final ranking (for Recall@K, MRR, nDCG)
    confidence: float = 0.0
    reason: str = ""
    rounds: int = 1
    rewrites: list = field(default_factory=list)
    trace: list = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    latency_s: float = 0.0
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class Pipeline:
    name = "base"

    def __init__(self, kb, llm, reranker=None):
        self.kb, self.llm, self.reranker = kb, llm, reranker

    def run(self, ticket: str) -> TicketResult:
        usage = Usage()
        t0 = time.perf_counter()
        result = TicketResult(arch=self.name, ticket=ticket)
        try:
            self._run(ticket, result, usage)
        except Exception as exc:  # noqa: BLE001 - API failure: escalate instead of crashing
            result.decision, result.error = "escalate", f"{type(exc).__name__}: {exc}"[:500]
            result.reason = "Lỗi hệ thống, chuyển nhân viên xử lý."
        result.latency_s = round(time.perf_counter() - t0, 3)
        result.usage = usage.to_dict()
        return result

    def _run(self, ticket: str, result: TicketResult, usage: Usage) -> None:
        raise NotImplementedError

    # shared: generate the answer from a list of chunks
    def generate(self, ticket: str, hits, result: TicketResult, usage: Usage, forced: bool = False) -> None:
        system, user, schema = answer_prompt(ticket, hits, forced=forced)
        data = self.llm.json("generate", system, user, schema, payload={"ticket": ticket, "hits": hits}, usage=usage)
        valid = {h.chunk.chunk_id for h in hits}
        result.decision = data.get("decision", "escalate")
        if result.decision not in ("answer", "clarify", "escalate"):
            result.decision = "escalate"
        result.reply = data.get("reply", "")
        result.handling_steps = data.get("handling_steps", [])
        result.clarifying_questions = data.get("clarifying_questions", [])
        result.citations = [c for c in data.get("citations", []) if c in valid]       # drop invented citations
        result.confidence = float(max(0.0, min(1.0, data.get("confidence", 0.0))))
        result.reason = data.get("reason", "")
        result.trace.append({"step": "generate", "decision": result.decision, "citations": result.citations})

