"""
Streamlit console for production-rag.

This is a second frontend for the SAME API the RAG Console artifact talks
to -- same endpoints, same requests, just a Python UI you run locally
instead of a page in the browser. Good for watching the pipeline from
inside a REPL-ish tool, or if you'd rather read/extend this in Python than
in JavaScript.

Run it with:
    pip install -r requirements-streamlit.txt
    streamlit run frontend/streamlit_app.py

It talks to your running FastAPI server over plain HTTP via `requests` --
no CORS involved (CORS is a BROWSER rule; this is a Python process calling
another process directly, same as curl), so you do NOT need the
CORS_ALLOW_ORIGINS change the browser console needed.
"""

import time
from datetime import datetime, timezone

import requests
import streamlit as st

st.set_page_config(page_title="RAG Console (Streamlit)", page_icon="🔎", layout="wide")

# ---------------------------------------------------------------------------
# Session state: everything that needs to survive a rerun (Streamlit reruns
# this whole script top-to-bottom on every interaction) lives here. This
# mirrors the browser console's localStorage -- it's per-session, not
# server-side truth; the server itself has no "list documents" endpoint.
# ---------------------------------------------------------------------------
if "docs" not in st.session_state:
    st.session_state.docs = []  # [{document_id, filename, page_count, indexed, chunk_count, cost_usd}]
if "base_url" not in st.session_state:
    st.session_state.base_url = "http://localhost:8000"
if "api_key" not in st.session_state:
    st.session_state.api_key = ""


def api(method: str, path: str, **kwargs):
    """One place every HTTP call goes through, so auth + base URL + error
    handling is consistent everywhere below -- same role apiFetch() plays
    in the browser console."""
    base = st.session_state.base_url.rstrip("/")
    headers = kwargs.pop("headers", {}) or {}
    if st.session_state.api_key:
        headers["Authorization"] = f"Bearer {st.session_state.api_key}"
    resp = requests.request(method, base + path, headers=headers, timeout=60, **kwargs)
    if not resp.ok:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text or f"HTTP {resp.status_code}"
        raise RuntimeError(detail if isinstance(detail, str) else str(detail))
    return resp.json() if resp.content else None


def usd(n: float) -> str:
    return f"${n:,.6f}"


# ---------------------------------------------------------------------------
# Sidebar: connection settings, same role as the browser console's ⚙ panel
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🔎 RAG Console")
    st.caption("Streamlit frontend for your production-rag API")

    st.session_state.base_url = st.text_input("API base URL", value=st.session_state.base_url)
    st.session_state.api_key = st.text_input(
        "API key (Bearer token)", value=st.session_state.api_key, type="password",
        help="Printed in your server's startup logs, or generate one with scripts/manage_keys.py",
    )

    if st.button("Test connection", use_container_width=True):
        try:
            health = requests.get(st.session_state.base_url.rstrip("/") + "/health", timeout=5).json()
            st.success(f"Connected -- {health}")
        except Exception as e:
            st.error(f"Could not reach the API: {e}")

    st.divider()
    st.subheader("Tracked documents")
    if not st.session_state.docs:
        st.caption("None yet. Upload one in the Upload tab.")
    else:
        for d in st.session_state.docs:
            status = f"{d['chunk_count']} chunks" if d["indexed"] else "not indexed"
            st.markdown(f"**{d['filename']}**  \n{d['page_count']} pages · {status}")
            st.caption(d["document_id"])

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_upload, tab_ask, tab_search, tab_chunks, tab_monitor = st.tabs(
    ["📤 Upload & Index", "💬 Ask", "🔍 Search", "📄 Chunks", "📊 Monitor"]
)

# ---- Upload & Index --------------------------------------------------------
with tab_upload:
    st.subheader("Upload a PDF, then index it")
    st.caption("Upload parses the PDF (free). Index calls OpenAI to embed the chunks (a fraction of a cent).")

    uploaded = st.file_uploader("PDF file", type=["pdf"])
    col1, col2 = st.columns([1, 3])
    with col1:
        dry_run = st.checkbox("Dry run (preview cost, don't actually embed)", value=False)

    if uploaded is not None and st.button("Upload & Index", type="primary"):
        try:
            with st.spinner("Uploading and parsing…"):
                parsed = api(
                    "POST", "/documents/upload",
                    files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                )
            doc_id = parsed["document_id"]
            st.success(f"Parsed {parsed['metadata']['page_count']} pages. document_id: `{doc_id}`")

            with st.expander("Page-by-page parse result", expanded=False):
                for p in parsed["pages"]:
                    st.markdown(f"**Page {p['page_number']}** · status: `{p['status']}` · {p['char_count']} chars")
                    if p["text"]:
                        st.text(p["text"][:400] + ("…" if len(p["text"]) > 400 else ""))

            # Track the document as soon as it's parsed -- even if indexing
            # fails next (e.g. no OPENAI_API_KEY configured yet), it still
            # shows up below so indexing can be retried without re-uploading.
            doc_record = {
                "document_id": doc_id,
                "filename": parsed["metadata"]["filename"],
                "page_count": parsed["metadata"]["page_count"],
                "indexed": False,
                "chunk_count": 0,
                "cost_usd": 0.0,
            }
            st.session_state.docs.insert(0, doc_record)

            with st.spinner("Indexing (embedding chunks via OpenAI)…"):
                report = api("POST", f"/documents/{doc_id}/index?dry_run={'true' if dry_run else 'false'}")

            doc_record["indexed"] = not dry_run
            doc_record["chunk_count"] = report["chunk_count"]
            doc_record["cost_usd"] = report["cost_usd"]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Chunks", report["chunk_count"])
            c2.metric("Tokens", report["tokens"])
            c3.metric("Cost", usd(report["cost_usd"]))
            c4.metric("Duration", f"{report['duration_ms']:.0f} ms")
            if report.get("warnings"):
                for w in report["warnings"]:
                    st.warning(w)
        except Exception as e:
            st.error(f"Failed: {e}")
            st.caption("If the document parsed but indexing failed, it's still listed in the sidebar -- fix the issue (e.g. set OPENAI_API_KEY) and use the Chunks tab or re-run indexing via curl/Swagger with its document_id.")

# ---- Ask --------------------------------------------------------------------
with tab_ask:
    st.subheader("Ask a grounded, cited question")

    doc_options = {"All documents": None}
    doc_options.update({f"{d['filename']} ({d['document_id'][:8]}…)": d["document_id"] for d in st.session_state.docs})
    scope_label = st.selectbox("Scope", list(doc_options.keys()), key="ask_scope")

    col1, col2 = st.columns(2)
    with col1:
        top_k = st.slider("Top K", 1, 10, 5)
    with col2:
        rerank = st.checkbox("Rerank", value=True)

    question = st.text_area("Question", placeholder="e.g. Which product has the highest gross margin?")

    if st.button("Ask", type="primary") and question.strip():
        body = {"question": question.strip(), "top_k": top_k, "rerank": rerank}
        if doc_options[scope_label]:
            body["document_id"] = doc_options[scope_label]
        try:
            with st.spinner("Retrieving → reranking → generating… (a cold free-tier server can take ~1 min to wake up)"):
                res = api("POST", "/ask", json=body)

            if res["insufficient_evidence"]:
                st.warning(f"Refused to answer ({res.get('refusal_reason') or 'insufficient evidence'}).")
            elif res["degraded"]:
                st.warning("Answered, but one step degraded gracefully -- see warnings below.")
            else:
                st.success(f"Grounded answer with {len(res['citations'])} citation(s).")

            st.markdown("#### Answer")
            st.write(res["answer"])

            if res["citations"]:
                st.markdown("#### Citations")
                for c in res["citations"]:
                    score = f" · rerank score {c['rerank_score']:.2f}" if c.get("rerank_score") is not None else ""
                    with st.expander(f"{c['source_id']} · page {c['page_number']}{score}"):
                        st.write(c["snippet"])

            for w in res.get("warnings", []):
                st.warning(w)

            # ---- Pipeline trace: the same idea as the browser console's
            # trace card -- show every stage that ran, its time, its cost,
            # and which optional stages were skipped and why.
            st.markdown("#### Pipeline trace")
            timings = res.get("timings_ms", {})
            cost = res["cost"]
            total_ms = timings.get("total", 0)

            timed_stages = [("retrieval", "Retrieve"), ("rerank", "Rerank"), ("generation", "Generate")]
            for key, label in timed_stages:
                if key in timings:
                    ms = timings[key]
                    frac = (ms / total_ms) if total_ms else 0
                    tcol1, tcol2, tcol3 = st.columns([1, 4, 1])
                    tcol1.write(f"**{label}**")
                    tcol2.progress(min(1.0, frac))
                    tcol3.write(f"{ms:.0f} ms")

            optional_stages = [
                ("contextualizer_cost_usd", "contextualizer_prompt_tokens", "contextualizer_completion_tokens",
                 "Contextualize", "no conversation_id"),
                ("memory_extraction_cost_usd", "memory_extraction_prompt_tokens", "memory_extraction_completion_tokens",
                 "Memory extract", "nothing durable found / refused"),
                ("memory_embedding_cost_usd", "memory_embedding_tokens", None,
                 "Memory embed", "no memories to embed"),
            ]
            for cost_key, tok_key, tok_key2, label, skip_reason in optional_stages:
                ran = cost.get(cost_key, 0) > 0
                tcol1, tcol2, tcol3 = st.columns([1, 4, 1])
                if ran:
                    tokens = cost.get(tok_key, 0) + (cost.get(tok_key2, 0) if tok_key2 else 0)
                    tcol1.write(f"**{label}**")
                    tcol2.caption(f"ran · {tokens} tok (no timing captured for this stage)")
                    tcol3.write(usd(cost[cost_key]))
                else:
                    tcol1.write(f"_{label}_")
                    tcol2.caption(f"skipped -- {skip_reason}")
                    tcol3.write("—")

            st.markdown("#### Cost breakdown")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total", usd(cost["total_cost_usd"]))
            c2.metric("Embedding", f"{cost['embedding_tokens']} tok")
            c3.metric("Rerank", f"{cost['rerank_prompt_tokens'] + cost['rerank_completion_tokens']} tok")
            c4.metric("Generation", f"{cost['generation_prompt_tokens'] + cost['generation_completion_tokens']} tok")

            with st.expander("Raw response JSON"):
                st.json(res)
        except Exception as e:
            st.error(f"Failed: {e}")

# ---- Search -------------------------------------------------------------
with tab_search:
    st.subheader("Keyword / vector / hybrid search")

    doc_options_s = {"All documents": None}
    doc_options_s.update({f"{d['filename']} ({d['document_id'][:8]}…)": d["document_id"] for d in st.session_state.docs})

    col1, col2, col3 = st.columns([3, 1, 2])
    with col1:
        query = st.text_input("Query")
    with col2:
        mode = st.selectbox("Mode", ["hybrid", "vector", "keyword"])
    with col3:
        scope_label_s = st.selectbox("Scope", list(doc_options_s.keys()), key="search_scope")

    if st.button("Search", type="primary") and query.strip():
        body = {"query": query.strip(), "mode": mode, "top_k": 10}
        if doc_options_s[scope_label_s]:
            body["document_id"] = doc_options_s[scope_label_s]
        try:
            with st.spinner("Searching…"):
                res = api("POST", "/search", json=body)
            st.caption(f"{len(res['hits'])} hit(s) · {res['score_type']} scoring" + (" · degraded to keyword-only" if res.get("degraded") else ""))
            for h in res["hits"]:
                extra = []
                if h.get("keyword_score") is not None:
                    extra.append(f"bm25 {h['keyword_score']:.2f}")
                if h.get("vector_score") is not None:
                    extra.append(f"cos {h['vector_score']:.3f}")
                with st.expander(f"#{h['rank']} · page {h['page_number']} · score {h['score']:.3f}" + (" · " + ", ".join(extra) if extra else "")):
                    st.write(h["text"])
        except Exception as e:
            st.error(f"Failed: {e}")

# ---- Chunks ---------------------------------------------------------------
with tab_chunks:
    st.subheader("Inspect a document's stored chunks")

    doc_options_c = {d["document_id"]: d["filename"] for d in st.session_state.docs}
    if not doc_options_c:
        st.caption("Upload and index a document first.")
    else:
        selected = st.selectbox(
            "Document", list(doc_options_c.keys()), format_func=lambda k: doc_options_c[k],
        )
        if st.button("Load chunks"):
            try:
                res = api("GET", f"/documents/{selected}/chunks?limit=200&offset=0")
                st.caption(f"{res['total']} total chunks (showing {len(res['chunks'])})")
                rows = [
                    {
                        "index": c["chunk_index"], "page": c["page_number"], "tokens": c["token_count"],
                        "embedded": c["has_embedding"],
                        "text": (c["text"][:140] + "…") if len(c["text"]) > 140 else c["text"],
                    }
                    for c in res["chunks"]
                ]
                st.dataframe(rows, use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"Failed: {e}")

# ---- Monitor ----------------------------------------------------------------
with tab_monitor:
    st.subheader("Live process stats")
    st.caption("GET /observability/summary -- this server's own in-process counters since it last started.")

    auto = st.checkbox("Auto-refresh every 5s", value=False, key="monitor_auto")
    refresh_clicked = st.button("Refresh now")

    if auto or refresh_clicked or "monitor_last" not in st.session_state:
        try:
            st.session_state.monitor_data = api("GET", "/observability/summary")
            st.session_state.monitor_last = datetime.now(timezone.utc)
        except Exception as e:
            st.error(f"Could not reach the API: {e}")

    data = st.session_state.get("monitor_data")
    if data:
        c1, c2, c3 = st.columns(3)
        c1.metric("Requests", data["requests"]["total"])
        c2.metric("Error rate", f"{data['requests']['error_rate'] * 100:.1f}%")
        c3.metric("Cache hit ratio", f"{data['cache']['hit_ratio'] * 100:.1f}%")
        c4, c5, c6 = st.columns(3)
        c4.metric("p50 latency", f"{data['requests']['latency_ms']['p50']:.0f} ms")
        c5.metric("p95 latency", f"{data['requests']['latency_ms']['p95']:.0f} ms")
        c6.metric("Total spend", usd(data["cost"]["total_usd"]))

        st.markdown("#### Cost by stage")
        stages = data["cost"]["usd_by_stage"]
        if not stages:
            st.caption("No paid requests yet.")
        else:
            rows = [
                {"stage": s, "tokens": data["cost"]["tokens_by_stage"].get(s, 0), "cost_usd": stages[s]}
                for s in stages
            ]
            st.dataframe(rows, use_container_width=True, hide_index=True)

        st.markdown("#### Refusals")
        reasons = data["refusals_by_reason"]
        if not reasons:
            st.caption("No refusals recorded yet.")
        else:
            st.dataframe(
                [{"reason": r, "count": n} for r, n in reasons.items()],
                use_container_width=True, hide_index=True,
            )

        st.caption(f"Last updated {st.session_state.monitor_last.strftime('%H:%M:%S UTC')}")

    if auto:
        time.sleep(5)
        st.rerun()
