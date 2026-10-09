"""OpenAI client with a stub (no network): strict json_schema, cache, token usage, retries, out-of-quota."""

import types as pytypes

import httpx
import numpy as np
import openai
import pytest

NS = pytypes.SimpleNamespace


def _rate_limit(msg="Rate limit reached", code="rate_limit_exceeded"):
    resp = httpx.Response(429, request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"))
    return openai.RateLimitError(msg, response=resp, body={"error": {"code": code}})


class _Client:
    def __init__(self, errors=()):
        self.calls, self.errors, self.emb_calls = [], list(errors), []
        self.chat = NS(completions=NS(create=self._create))
        self.embeddings = NS(create=self._embed)

    def _create(self, **kw):
        self.calls.append(kw)
        if self.errors:
            raise self.errors.pop(0)
        msg = NS(content='{"verdict": "sufficient", "relevant_ids": ["a"]}')
        return NS(choices=[NS(message=msg)], usage=NS(prompt_tokens=200, completion_tokens=40))

    def _embed(self, model, input, dimensions):
        self.emb_calls.append(list(input))
        data = [NS(index=i, embedding=[1.0, float(i), 0.5]) for i in range(len(input))]
        return NS(data=list(reversed(data)))           # returned out of order -> must sort by index


@pytest.fixture
def patch_openai(monkeypatch):
    from rag_cskh import llm as llm_mod
    from rag_cskh.config import settings
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")
    monkeypatch.setattr(llm_mod.time, "sleep", lambda s: None)

    def install(client):
        monkeypatch.setattr(openai, "OpenAI", lambda **kw: client)
        return client
    return install


def test_strict_schema_closes_every_object():
    from rag_cskh.llm import strict_schema
    from rag_cskh.prompts import ANALYZE_SCHEMA, ANSWER_SCHEMA, GRADE_SCHEMA, JUDGE_SCHEMA, RERANK_SCHEMA

    def check(s):
        if isinstance(s, dict):
            if s.get("type") == "object" and "properties" in s:
                assert s["additionalProperties"] is False and set(s["required"]) == set(s["properties"])
            for v in s.values():
                check(v)
        elif isinstance(s, list):
            for v in s:
                check(v)
    for schema in (ANALYZE_SCHEMA, GRADE_SCHEMA, ANSWER_SCHEMA, RERANK_SCHEMA, JUDGE_SCHEMA):
        check(strict_schema(schema))


def test_openai_llm_strict_output_cache_and_retry(patch_openai, tmp_path):
    from rag_cskh.llm import OpenAILLM, Usage
    from rag_cskh.prompts import GRADE_SCHEMA
    client = patch_openai(_Client(errors=[_rate_limit()]))
    llm = OpenAILLM(model="gpt-4o-mini", rpm=6000, cache_dir=tmp_path)
    u = Usage()
    out = llm.json("grade", "system", "user", GRADE_SCHEMA, usage=u)
    assert out["verdict"] == "sufficient"
    assert len(client.calls) == 2                       # first call 429 -> retried
    kw = client.calls[-1]
    assert kw["model"] == "gpt-4o-mini" and kw["temperature"] == 0.0
    assert kw["messages"][0] == {"role": "system", "content": "system"}
    rf = kw["response_format"]
    assert rf["type"] == "json_schema" and rf["json_schema"]["strict"] is True and rf["json_schema"]["name"] == "grade"
    assert rf["json_schema"]["schema"]["additionalProperties"] is False
    assert (u.calls, u.input_tokens, u.output_tokens, u.cached_calls) == (1, 200, 40, 0)
    again = llm.json("grade", "system", "user", GRADE_SCHEMA, usage=u)   # cache hit: no API call
    assert again == out and len(client.calls) == 2 and u.cached_calls == 1


def test_openai_insufficient_quota_is_not_retried(patch_openai, tmp_path):
    from rag_cskh.llm import LLMError, OpenAILLM
    from rag_cskh.prompts import GRADE_SCHEMA
    err = _rate_limit("You exceeded your current quota", code="insufficient_quota")
    err = openai.RateLimitError("Error code: 429 - insufficient_quota", response=err.response, body=err.body)
    client = patch_openai(_Client(errors=[err]))
    llm = OpenAILLM(model="gpt-4o-mini", rpm=6000, cache_dir=tmp_path)
    with pytest.raises(LLMError, match="insufficient_quota"):
        llm.json("grade", "system", "user", GRADE_SCHEMA)
    assert len(client.calls) == 1


def test_openai_embedder_batches_orders_and_caches(patch_openai, tmp_path):
    from rag_cskh.embeddings import OpenAIEmbedder
    client = patch_openai(_Client())
    emb = OpenAIEmbedder(model="text-embedding-3-small", dim=3, cache_dir=tmp_path)
    m = emb.embed_documents(["a", "b", "c"], batch=2)
    assert m.shape == (3, 3) and np.allclose(np.linalg.norm(m, axis=1), 1)
    assert client.emb_calls == [["a", "b"], ["c"]]
    assert m[1, 1] > m[0, 1]                            # order follows index
    emb2 = OpenAIEmbedder(model="text-embedding-3-small", dim=3, cache_dir=tmp_path)
    assert len(emb2.cache) == 3
    emb2.embed_documents(["a", "b", "c"])
    assert len(client.emb_calls) == 2                   # no new API call
    assert emb.embed_query("x").shape == (3,)
