"""LLM clients that return JSON matching a schema.

  OpenAILLM  OpenAI Chat Completions with Structured Outputs (strict json_schema), request-per-minute limit,
             retries on 429/5xx, disk cache keyed by (model, prompt, schema) so re-running the evaluation is free.
  FakeLLM    simple deterministic rules, no network - used by the offline tests.

Every call is recorded in Usage: calls, input/output tokens and latency (for the performance and cost metrics).
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .config import settings


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    cached_calls: int = 0
    by_task: Counter = field(default_factory=Counter)

    def add(self, task: str, inp: int, out: int, latency: float, cached: bool) -> None:
        self.calls += 1
        self.input_tokens += inp
        self.output_tokens += out
        self.latency_s += latency
        self.cached_calls += int(cached)
        self.by_task[task] += 1

    @property
    def cost_usd(self) -> float:
        return self.input_tokens / 1e6 * settings.price_in_per_m + self.output_tokens / 1e6 * settings.price_out_per_m

    def to_dict(self) -> dict:
        return {"calls": self.calls, "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                "llm_latency_s": round(self.latency_s, 3), "cached_calls": self.cached_calls,
                "cost_usd": round(self.cost_usd, 6), "by_task": dict(self.by_task)}


class LLMError(RuntimeError):
    pass


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    return json.loads(text)


class CachedLLM:
    """Disk cache keyed by (model, prompt, schema), request-per-minute limit and retries."""

    provider = "base"
    _lock = threading.Lock()
    _last_call = 0.0

    def __init__(self, model: str, rpm: int | None = None, cache_dir: Path | None = None):
        self.model = model
        self.interval = 60.0 / max(1, rpm or settings.llm_rpm)
        self.cache_dir = (cache_dir or settings.cache_dir) / "llm"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _wait_turn(self) -> None:
        with CachedLLM._lock:
            now = time.monotonic()
            wait = CachedLLM._last_call + self.interval - now
            if wait > 0:
                time.sleep(wait)
            CachedLLM._last_call = time.monotonic()

    def _call(self, system: str, user: str, schema: dict, temperature: float, task: str) -> tuple[str, int, int]:
        """-> (json text, input tokens, output tokens)"""
        raise NotImplementedError

    def _retryable(self, exc: Exception) -> bool:
        raise NotImplementedError

    def json(self, task: str, system: str, user: str, schema: dict, payload: dict | None = None,
             usage: Usage | None = None, temperature: float = 0.0) -> dict:
        key = hashlib.sha256(json.dumps([self.provider, self.model, system, user, schema, temperature],
                                        ensure_ascii=False).encode()).hexdigest()
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            rec = json.loads(path.read_text(encoding="utf-8"))
            if usage is not None:
                usage.add(task, rec["input_tokens"], rec["output_tokens"], rec["latency_s"], cached=True)
            return rec["data"]
        last_exc = None
        for attempt in range(6):
            self._wait_turn()
            t0 = time.perf_counter()
            try:
                text, inp, out = self._call(system, user, schema, temperature, task)
                latency = time.perf_counter() - t0
                data = _parse_json(text)
                path.write_text(json.dumps({"task": task, "data": data, "input_tokens": inp, "output_tokens": out,
                                            "latency_s": latency}, ensure_ascii=False), encoding="utf-8")
                if usage is not None:
                    usage.add(task, inp, out, latency, cached=False)
                return data
            except (json.JSONDecodeError, ValueError) as exc:          # malformed JSON: retry
                last_exc = exc
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if not self._retryable(exc):
                    raise LLMError(f"{task}: {str(exc)[:300]}") from exc
                msg = str(exc)
                time.sleep(min(60, 10 * (attempt + 1)) if "429" in msg or "rate" in msg.lower()
                           else 2 ** attempt)
        raise LLMError(f"{task}: retries exhausted ({last_exc})")


def strict_schema(schema: dict) -> dict:
    """Make a JSON schema OpenAI-strict: every object gets additionalProperties=false and all keys required."""
    if isinstance(schema, dict):
        out = {k: strict_schema(v) for k, v in schema.items()}
        if out.get("type") == "object" and "properties" in out:
            out["additionalProperties"] = False
            out["required"] = list(out["properties"])
        return out
    if isinstance(schema, list):
        return [strict_schema(v) for v in schema]
    return schema


class OpenAILLM(CachedLLM):
    """OpenAI Chat Completions + Structured Outputs (json_schema, strict)."""

    provider = "openai"

    def __init__(self, model: str | None = None, rpm: int | None = None, cache_dir: Path | None = None):
        from openai import OpenAI
        if not settings.openai_api_key:
            raise LLMError("OPENAI_API_KEY is not set (.env)")
        super().__init__(model or settings.llm_model, rpm, cache_dir)
        self.client = OpenAI(api_key=settings.openai_api_key, max_retries=0, timeout=90)

    def _call(self, system, user, schema, temperature, task):
        resp = self.client.chat.completions.create(
            model=self.model, temperature=temperature,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={"type": "json_schema",
                             "json_schema": {"name": task, "strict": True, "schema": strict_schema(schema)}})
        u = resp.usage
        return resp.choices[0].message.content, int(u.prompt_tokens), int(u.completion_tokens)

    def _retryable(self, exc):
        import openai
        if "insufficient_quota" in str(exc):          # out of credit: stop immediately
            return False
        return isinstance(exc, (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError,
                                openai.InternalServerError))


# --------------------------------------------------------------------------- #
# Fake LLM (offline)
# --------------------------------------------------------------------------- #
class FakeLLM:
    """Rule-based stand-in for the LLM, used only to test the control flow offline."""

    model = "fake"

    def json(self, task: str, system: str, user: str, schema: dict, payload: dict | None = None,
             usage: Usage | None = None, temperature: float = 0.0) -> dict:
        payload = payload or {}
        handler = getattr(self, f"_{task}")
        data = handler(payload)
        if usage is not None:
            usage.add(task, len(system + user) // 4, len(json.dumps(data, ensure_ascii=False)) // 4, 0.0, cached=False)
        return data

    @staticmethod
    def _overlap(a: str, b: str) -> float:
        from .text import bm25_tokens
        ta, tb = set(t for t in bm25_tokens(a) if "_" not in t), set(bm25_tokens(b))
        return len(ta & tb) / max(1, len(ta))

    def _analyze(self, p):
        t = p["ticket"]
        vague = len(t.split()) <= 9 and not re.search(r"(sora|kiwa|lumio|vinto|haru|nexa|polar|zenko|ferro|printo|tivi|tủ lạnh|điều hòa|máy giặt|laptop|điện thoại)", t.lower())
        return {"intent": "khác", "product": None, "entities": [], "missing_info": ["sản phẩm"] if vague else [],
                "search_queries": [t]}

    def _grade(self, p):
        hits, t = p["hits"], p["ticket"]
        if p.get("analysis", {}).get("missing_info"):
            return {"verdict": "missing_info", "relevant_ids": [], "conflict_ids": [], "missing_info": p["analysis"]["missing_info"],
                    "rewritten_query": "", "explanation": "missing info", "confidence": 0.6}
        scored = [(self._overlap(t, h.chunk.search_text), h) for h in hits]
        good = [h.chunk.chunk_id for s, h in scored if s >= 0.35 and h.chunk.status == "active"]
        if good:
            return {"verdict": "sufficient", "relevant_ids": good[:4], "conflict_ids": [], "missing_info": [],
                    "rewritten_query": "", "explanation": "enough", "confidence": 0.7}
        if p["round"] < p["max_rounds"]:
            return {"verdict": "insufficient", "relevant_ids": [], "conflict_ids": [], "missing_info": [],
                    "rewritten_query": t + " chính sách quy định", "explanation": "not enough", "confidence": 0.4}
        return {"verdict": "no_evidence", "relevant_ids": [], "conflict_ids": [], "missing_info": [], "rewritten_query": "",
                "explanation": "no evidence", "confidence": 0.5}

    def _generate(self, p):
        hits = p["hits"]
        if not hits:
            return {"decision": "escalate", "reply": "", "handling_steps": [], "clarifying_questions": [], "citations": [],
                    "confidence": 0.3, "reason": "no documents"}
        top = hits[0].chunk
        return {"decision": "answer", "reply": f"Dạ, theo {top.title}: {top.text}", "handling_steps": ["Phản hồi khách theo chính sách"],
                "clarifying_questions": [], "citations": [top.chunk_id], "confidence": 0.6, "reason": ""}

    def _judge(self, p):
        ans = p["answer"].lower()
        from .eval.metrics import norm_fact
        ok = all(norm_fact(f) in norm_fact(ans) for f in p["key_facts"]) if p["key_facts"] else True
        return {"correct": ok, "faithfulness": 1.0 if p["evidence"] else 0.0, "unsupported_claims": [], "citations_support": bool(p["evidence"])}

    def _rerank(self, p):
        return {"scores": [{"id": h.chunk.chunk_id, "score": round(10 * self._overlap(p["query"], h.chunk.search_text), 2)} for h in p["hits"]]}


def get_llm(provider: str | None = None, model: str | None = None):
    provider = provider or settings.llm_provider
    if provider == "openai":
        return OpenAILLM(model=model)
    if provider == "fake":
        return FakeLLM()
    raise ValueError(f"unknown LLM_PROVIDER {provider}")
