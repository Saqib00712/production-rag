"""
Day 16: `contextualize`, `memory_retrieve`, and `memory_extract` now get
their own `tracing.span()`s, closing the observability gap flagged (but
left open) in docs/day13.md and docs/day14.md. Spans land in the audit
log's consolidated per-request line (app/middleware/audit_log.py), so
these tests read them the same way test_observability_endpoints.py already
does for retrieval/rerank/generation: via `caplog` on the "audit" logger.
"""
import logging

from fpdf import FPDF


def _pdf(pages=("Refund policy. Customers may request a refund within thirty days of purchase.",)):
    pdf = FPDF()
    for t in pages:
        pdf.add_page(); pdf.set_font("Helvetica", size=12); pdf.multi_cell(0, 10, t)
    return bytes(pdf.output())


def _index(client):
    r = client.post("/documents/upload", files={"file": ("d.pdf", _pdf(), "application/pdf")})
    doc = r.json()["document_id"]
    assert client.post(f"/documents/{doc}/index").status_code == 200
    return doc


def _span_names(caplog) -> set[str]:
    audit_records = [r for r in caplog.records if r.name == "audit"]
    assert audit_records, "expected at least one audit log line"
    return {s["name"] for s in audit_records[-1].spans}


def test_contextualize_span_recorded_on_a_followup(ask_client, generator_factory, contextualizer_factory, caplog):
    doc = _index(ask_client)
    cid = ask_client.post("/conversations").json()["conversation_id"]
    ask_client.post("/ask", json={"question": "refund policy", "document_id": doc, "conversation_id": cid})

    caplog.set_level(logging.INFO)
    caplog.clear()
    ask_client.post("/ask", json={"question": "what about the second one?", "document_id": doc, "conversation_id": cid})
    assert "contextualize" in _span_names(caplog)


def test_contextualize_span_still_recorded_but_free_on_a_conversations_first_turn(ask_client, generator_factory, caplog):
    """The span wraps the ATTEMPT (so a trace always shows contextualization
    was considered for a conversation_id call), even though the contextualizer
    itself skips the LLM call internally when there's no history to resolve
    -- that's a cost fact (see test_cost_accounting.py), not a tracing one."""
    doc = _index(ask_client)
    cid = ask_client.post("/conversations").json()["conversation_id"]
    caplog.set_level(logging.INFO)
    caplog.clear()
    ask_client.post("/ask", json={"question": "refund policy", "document_id": doc, "conversation_id": cid})
    assert "contextualize" in _span_names(caplog)


def test_no_contextualize_span_without_a_conversation_id_at_all(ask_client, generator_factory, caplog):
    doc = _index(ask_client)
    caplog.set_level(logging.INFO)
    caplog.clear()
    ask_client.post("/ask", json={"question": "refund policy", "document_id": doc})
    assert "contextualize" not in _span_names(caplog)


def test_memory_retrieve_span_recorded_when_memories_exist(ask_client, generator_factory, caplog):
    ask_client.post("/memories", json={"content": "Prefers answers in metric units."})
    doc = _index(ask_client)
    caplog.set_level(logging.INFO)
    caplog.clear()
    ask_client.post("/ask", json={"question": "refund policy", "document_id": doc})
    assert "memory_retrieve" in _span_names(caplog)


def test_memory_extract_span_recorded_after_a_real_answer(ask_client, generator_factory, memory_extractor_factory, caplog):
    memory_extractor_factory.facts = ["Works in procurement."]
    doc = _index(ask_client)
    caplog.set_level(logging.INFO)
    caplog.clear()
    ask_client.post("/ask", json={"question": "refund policy", "document_id": doc})
    assert "memory_extract" in _span_names(caplog)


def test_no_extra_spans_when_there_is_nothing_extra_to_do(ask_client, generator_factory, caplog):
    # No conversation_id, no stored memories, no document-indexed context
    # for extraction to key off of beyond the answer itself -- none of the
    # three Day 13/14/15 extras should fire here.
    doc = _index(ask_client)
    caplog.set_level(logging.INFO)
    caplog.clear()
    ask_client.post("/ask", json={"question": "refund policy", "document_id": doc})
    names = _span_names(caplog)
    assert "contextualize" not in names
    assert "memory_retrieve" not in names
    # memory_extract DOES still fire here (a real answer with the default
    # ScriptedMemoryExtractorFactory, which returns no facts) -- that's
    # correct: extraction is attempted whenever there's a real answer,
    # independent of whether memories already exist. Covered by the
    # dedicated extract test above; not asserted away here.
