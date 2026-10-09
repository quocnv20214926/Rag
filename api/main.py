"""FastAPI service: process a support ticket with one of the three RAG architectures.

    uvicorn api.main:app --reload --port 8000
    POST /tickets/process   {"ticket": "...", "arch": "corrective_rag"}
    POST /tickets/compare   {"ticket": "..."}          -> results of all three architectures
    GET  /chunks/{chunk_id}                            -> text of a cited chunk
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from rag_cskh.service import describe, get_service  # noqa: E402

app = FastAPI(title="Corrective RAG - customer-support assistant", version="1.0.0")
Arch = Literal["vector_rag", "hybrid_rag", "corrective_rag"]


class TicketIn(BaseModel):
    ticket: str = Field(..., min_length=3, max_length=4000, examples=["Tủ lạnh Kiwa mua 3 tháng bị hỏng, bảo hành mất bao lâu?"])
    arch: Arch = "corrective_rag"


class CompareIn(BaseModel):
    ticket: str = Field(..., min_length=3, max_length=4000)


def _with_sources(res: dict) -> dict:
    kb = get_service()["kb"]
    ids = list(dict.fromkeys(res["citations"] + res["context_ids"]))
    res["sources"] = [{"chunk_id": i, "title": kb.by_id[i].title, "section": kb.by_id[i].section_title,
                       "status": kb.by_id[i].status, "text": kb.by_id[i].text, "cited": i in res["citations"]}
                      for i in ids if i in kb.by_id]
    return res


@app.get("/health")
def health():
    svc = get_service()
    return {"status": "ok", "chunks": len(svc["kb"].chunks), **describe()}


@app.post("/tickets/process")
def process(body: TicketIn):
    return _with_sources(get_service()["pipelines"][body.arch].run(body.ticket).to_dict())


@app.post("/tickets/compare")
def compare(body: CompareIn):
    return {name: _with_sources(p.run(body.ticket).to_dict()) for name, p in get_service()["pipelines"].items()}


@app.get("/chunks/{chunk_id:path}")
def chunk(chunk_id: str):
    c = get_service()["kb"].by_id.get(chunk_id)
    if not c:
        raise HTTPException(404, "chunk not found")
    return {"chunk_id": c.chunk_id, "doc_id": c.doc_id, "title": c.title, "section": c.section_title,
            "status": c.status, "effective_date": c.effective_date, "text": c.text}
