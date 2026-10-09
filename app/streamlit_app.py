"""Agent UI: paste a ticket -> decision, handling steps, drafted reply, cited sources, confidence.

    streamlit run app/streamlit_app.py          (API_URL defaults to http://localhost:8000)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")
ROOT = Path(__file__).resolve().parents[1]
ARCH_LABEL = {"corrective_rag": "Corrective RAG", "hybrid_rag": "Hybrid RAG", "vector_rag": "Vector RAG"}
DECISION = {"answer": ("✅ Enough evidence - answer", "green"), "clarify": ("❓ Missing information - ask the customer", "orange"),
            "escalate": ("⛔ No conclusion - escalate to an agent", "red")}

st.set_page_config(page_title="Support assistant", page_icon="💬", layout="wide")


@st.cache_data
def sample_tickets():
    path = ROOT / "data" / "eval" / "tickets.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [f"[{r['type']}] {r['ticket']}" for r in rows]


def show_result(res: dict, compact: bool = False):
    label, color = DECISION.get(res["decision"], (res["decision"], "gray"))
    st.markdown(f"#### :{color}[{label}]")
    c1, c2, c3 = st.columns(3)
    c1.metric("Confidence", f"{res['confidence']:.0%}")
    c2.metric("Retrieval rounds", res["rounds"])
    c3.metric("Latency", f"{res['latency_s']:.1f}s")
    if res.get("error"):
        st.error(res["error"])
    if res["reason"]:
        st.caption(f"Reason: {res['reason']}")
    st.markdown("**Drafted reply to the customer**")
    st.info(res["reply"] or "(none)")
    if res["clarifying_questions"]:
        st.markdown("**Clarifying questions**")
        for q in res["clarifying_questions"]:
            st.markdown(f"- {q}")
    if res["handling_steps"]:
        st.markdown("**Handling steps for the agent**")
        for s in res["handling_steps"]:
            st.markdown(f"- {s}")
    if res.get("sources"):
        st.markdown("**Sources**")
        for s in res["sources"]:
            if compact and not s["cited"]:
                continue
            tag = "📌 cited" if s["cited"] else "context"
            expired = " · ⚠️ expired" if s["status"] != "active" else ""
            with st.expander(f"{tag} · {s['title']} › {s['section']}{expired}"):
                st.caption(s["chunk_id"])
                st.write(s["text"])
    if not compact:
        with st.expander("Trace"):
            if res["rewrites"]:
                st.write("Rewritten queries:", res["rewrites"])
            st.json(res["trace"])
            st.json(res["usage"])


st.title("💬 Corrective RAG support assistant")
st.caption("BM25 + vector retrieval, reranking, evidence grading, query rewriting, cited answers. "
           "All company data is fictional.")

tab1, tab2 = st.tabs(["Process a ticket", "Evaluation results"])
with tab1:
    samples = sample_tickets()
    pick = st.selectbox("Sample ticket (optional)", [""] + samples)
    ticket = st.text_area("Customer ticket", value=pick.split("] ", 1)[1] if pick else "", height=110)
    mode = st.radio("Mode", ["Corrective RAG", "Compare 3 architectures"], horizontal=True)
    if st.button("Run", type="primary", disabled=not ticket.strip()):
        try:
            if mode == "Corrective RAG":
                with st.spinner("Retrieving evidence and drafting a reply..."):
                    r = requests.post(f"{API}/tickets/process", json={"ticket": ticket, "arch": "corrective_rag"}, timeout=300)
                r.raise_for_status()
                show_result(r.json())
            else:
                with st.spinner("Running all three architectures..."):
                    r = requests.post(f"{API}/tickets/compare", json={"ticket": ticket}, timeout=600)
                r.raise_for_status()
                cols = st.columns(3)
                for col, (arch, res) in zip(cols, r.json().items()):
                    with col:
                        st.subheader(ARCH_LABEL[arch])
                        show_result(res, compact=True)
        except requests.RequestException as exc:
            st.error(f"Cannot reach the API at {API}: {exc}")

with tab2:
    summary = ROOT / "reports" / "runs" / "summary.md"
    chart = ROOT / "reports" / "runs" / "comparison.png"
    if chart.exists():
        st.image(str(chart))
    if summary.exists():
        st.markdown(summary.read_text(encoding="utf-8"))
    else:
        st.info("No results yet. Run: python -m rag_cskh.cli eval")
