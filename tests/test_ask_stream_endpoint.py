"""
HTTP-level tests for POST /ask/stream, parsing real Server-Sent Events off
a real (in-process) TestClient response stream.
"""

import json

import pytest
from fpdf import FPDF

from app.dependencies import get_streaming_generator_factory
from app.main import app
from tests.fakes import FakeStreamingGenerator


def _pdf(pages):
    pdf = FPDF()
    for t in pages:
        pdf.add_page(); pdf.set_font("Helvetica", size=12); pdf.multi_cell(0, 10, t)
    return bytes(pdf.output())


def _index(client, pages=("Refund policy. Customers may request a refund within thirty days.",)):
    r = client.post("/documents/upload", files={"file": ("d.pdf", _pdf(pages), "application/pdf")})
    doc = r.json()["document_id"]
    assert client.post(f"/documents/{doc}/index").status_code == 200
    return doc


def _parse_sse(text: str) -> list[dict]:
    events = []
    for block in text.strip().split("\n\n"):
        if not block.strip():
            continue
        lines = block.strip().split("\n")
        event_type = next(l.split(": ", 1)[1] for l in lines if l.startswith("event: "))
        data = next(l.split(": ", 1)[1] for l in lines if l.startswith("data: "))
        events.append({"event": event_type, "data": json.loads(data)})
    return events


def test_stream_endpoint_returns_deltas_then_done(ask_client, generator_factory):
    doc = _index(ask_client)
    sg = FakeStreamingGenerator(["Refunds ", "within 30 days [S1]."])
    app.dependency_overrides[get_streaming_generator_factory] = lambda: (lambda: sg)

    with ask_client.stream("POST", "/ask/stream", json={"question": "refund policy", "document_id": doc}) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        body = "".join(r.iter_text())

    events = _parse_sse(body)
    assert [e["event"] for e in events] == ["delta", "delta", "done"]
    assert events[0]["data"] == {"text": "Refunds "}
    done = events[-1]["data"]
    assert done["insufficient_evidence"] is False
    assert len(done["citations"]) == 1
    app.dependency_overrides.pop(get_streaming_generator_factory, None)


def test_stream_endpoint_refuses_cleanly_with_no_index(ask_client):
    sg = FakeStreamingGenerator(["never used"])
    app.dependency_overrides[get_streaming_generator_factory] = lambda: (lambda: sg)

    with ask_client.stream("POST", "/ask/stream", json={"question": "anything"}) as r:
        body = "".join(r.iter_text())

    events = _parse_sse(body)
    assert len(events) == 1 and events[0]["event"] == "done"
    assert events[0]["data"]["refusal_reason"] == "no_results"
    app.dependency_overrides.pop(get_streaming_generator_factory, None)


def test_stream_endpoint_emits_correction_for_ungrounded_answer(ask_client):
    doc = _index(ask_client)
    sg = FakeStreamingGenerator(["A confident answer with no citation at all."])
    app.dependency_overrides[get_streaming_generator_factory] = lambda: (lambda: sg)

    with ask_client.stream("POST", "/ask/stream", json={"question": "refund policy", "document_id": doc}) as r:
        body = "".join(r.iter_text())

    events = _parse_sse(body)
    assert [e["event"] for e in events] == ["delta", "correction", "done"]
    assert "message" in events[1]["data"]
    assert events[-1]["data"]["refusal_reason"] == "ungrounded_answer"
    app.dependency_overrides.pop(get_streaming_generator_factory, None)


def test_stream_endpoint_validation_error_before_streaming_starts(ask_client):
    r = ask_client.post("/ask/stream", json={"question": ""})
    assert r.status_code == 422  # Pydantic validation still runs before the stream opens


def test_stream_endpoint_is_rate_limited_same_as_ask(ask_client, settings):
    doc = _index(ask_client)  # do this BEFORE lowering the limit -- upload+index share the /documents bucket
    sg = FakeStreamingGenerator(["x [S1]"])
    app.dependency_overrides[get_streaming_generator_factory] = lambda: (lambda: sg)
    app.state.rate_limiter.limit = 1
    app.state.rate_limiter.reset()
    try:
        with ask_client.stream("POST", "/ask/stream", json={"question": "q", "document_id": doc}) as r1:
            "".join(r1.iter_text())
        with ask_client.stream("POST", "/ask/stream", json={"question": "q2", "document_id": doc}) as r2:
            assert r2.status_code == 429
    finally:
        app.state.rate_limiter.limit = 10_000
        app.state.rate_limiter.reset()
        app.dependency_overrides.pop(get_streaming_generator_factory, None)


def test_stream_endpoint_appends_a_turn_when_given_a_conversation_id(ask_client):
    doc = _index(ask_client)
    sg = FakeStreamingGenerator(["Refunds ", "within 30 days [S1]."])
    app.dependency_overrides[get_streaming_generator_factory] = lambda: (lambda: sg)

    cid = ask_client.post("/conversations").json()["conversation_id"]
    with ask_client.stream(
        "POST", "/ask/stream", json={"question": "refund policy", "document_id": doc, "conversation_id": cid}
    ) as r:
        assert r.status_code == 200
        "".join(r.iter_text())

    history = ask_client.get(f"/conversations/{cid}").json()["turns"]
    assert len(history) == 1
    assert history[0]["question"] == "refund policy"
    app.dependency_overrides.pop(get_streaming_generator_factory, None)


def test_stream_endpoint_with_unowned_conversation_id_is_404_before_streaming(ask_client):
    from app.auth import TenantContext, get_current_tenant

    cid = ask_client.post("/conversations").json()["conversation_id"]
    app.dependency_overrides[get_current_tenant] = lambda: TenantContext(tenant_id="someone-else", key_label="x")
    try:
        r = ask_client.post("/ask/stream", json={"question": "anything", "conversation_id": cid})
        assert r.status_code == 404
    finally:
        app.dependency_overrides.pop(get_current_tenant, None)


def test_stream_endpoint_passes_long_term_memory_notes_to_generation(ask_client):
    """Day 14: memory is tenant-scoped, so it's used on /ask/stream too,
    with no conversation_id required at all."""
    doc = _index(ask_client)
    ask_client.post("/memories", json={"content": "Prefers answers in metric units."})
    sg = FakeStreamingGenerator(["Refunds ", "within 30 days [S1]."])
    app.dependency_overrides[get_streaming_generator_factory] = lambda: (lambda: sg)
    try:
        with ask_client.stream("POST", "/ask/stream", json={"question": "refund policy", "document_id": doc}) as r:
            "".join(r.iter_text())
        assert sg.last_memory_notes == ["Prefers answers in metric units."]
    finally:
        app.dependency_overrides.pop(get_streaming_generator_factory, None)


def test_stream_endpoint_extracts_a_durable_fact_after_a_real_answer(ask_client, memory_extractor_factory):
    doc = _index(ask_client)
    memory_extractor_factory.facts = ["Works in procurement."]
    sg = FakeStreamingGenerator(["Refunds ", "within 30 days [S1]."])
    app.dependency_overrides[get_streaming_generator_factory] = lambda: (lambda: sg)
    try:
        with ask_client.stream("POST", "/ask/stream", json={"question": "refund policy", "document_id": doc}) as r:
            "".join(r.iter_text())
    finally:
        app.dependency_overrides.pop(get_streaming_generator_factory, None)

    stored = ask_client.get("/memories").json()["memories"]
    assert [m["content"] for m in stored] == ["Works in procurement."]
