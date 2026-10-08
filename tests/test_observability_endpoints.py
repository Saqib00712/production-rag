"""
Integration tests for /metrics, /observability/summary, /observability/dashboard,
plus the key Day 9 promise: one request_id now flows through the audit log
AND every internal service log for the same HTTP request.
"""

from fpdf import FPDF


def _pdf(text="Refund policy. Customers may request a refund within thirty days."):
    pdf = FPDF()
    pdf.add_page(); pdf.set_font("Helvetica", size=12); pdf.multi_cell(0, 10, text)
    return bytes(pdf.output())


def test_metrics_endpoint_returns_prometheus_text(client):
    client.get("/health")
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "text/plain" in r.headers["content-type"]
    assert "http_requests_total" in r.text


def test_metrics_endpoint_is_not_rate_limited(client, settings):
    client.get("/health")
    from app.main import app

    app.state.rate_limiter.limit = 1
    app.state.rate_limiter.reset()
    try:
        for _ in range(5):
            assert client.get("/metrics").status_code == 200
    finally:
        app.state.rate_limiter.limit = 10_000
        app.state.rate_limiter.reset()


def test_summary_endpoint_reflects_recorded_requests(client):
    for _ in range(3):
        client.get("/health")
    body = client.get("/observability/summary").json()
    assert body["requests"]["total"] >= 3
    assert "p50" in body["requests"]["latency_ms"]


def test_summary_reflects_indexing_cost_and_cache_hits(client, embedder):
    doc = client.post("/documents/upload", files={"file": ("a.pdf", _pdf(), "application/pdf")}).json()["document_id"]
    client.post(f"/documents/{doc}/index")
    client.post(f"/documents/{doc}/index")  # second call is a full cache hit
    body = client.get("/observability/summary").json()
    assert body["cost"]["tokens_by_stage"].get("embedding", 0) > 0
    assert body["cache"]["hits"] >= 1


def test_summary_reflects_contextualizer_and_memory_costs(ask_client, generator_factory, memory_extractor_factory):
    """Day 17: before this, /observability/summary's cost totals only ever
    reflected embedding/rerank/generation -- a conversation or memory-
    bearing request spent real money (already counted in AskResponse.cost
    and the tenant budget since Day 16) that never showed up here."""
    memory_extractor_factory.facts = ["Prefers answers in metric units."]
    doc = _index(ask_client)
    cid = ask_client.post("/conversations").json()["conversation_id"]
    ask_client.post("/ask", json={"question": "refund policy", "document_id": doc, "conversation_id": cid})
    # Second turn: a real contextualizer rewrite happens (there's history),
    # AND a memory fact already exists for memory_retrieve to embed against.
    ask_client.post("/ask", json={"question": "what about it?", "document_id": doc, "conversation_id": cid})

    body = ask_client.get("/observability/summary").json()
    usd_by_stage = body["cost"]["usd_by_stage"]
    assert usd_by_stage.get("contextualize", 0) > 0
    assert usd_by_stage.get("memory_extract", 0) > 0
    assert usd_by_stage.get("memory_embedding", 0) > 0
    # total_usd must include these, not just embedding/rerank/generation
    assert body["cost"]["total_usd"] >= sum(usd_by_stage.values()) - 1e-9


def _index(client):
    doc = client.post("/documents/upload", files={"file": ("a.pdf", _pdf(), "application/pdf")}).json()["document_id"]
    client.post(f"/documents/{doc}/index")
    return doc


def test_summary_reflects_refusals(client):
    client.post("/ask", json={"question": "anything, nothing is indexed"})
    body = client.get("/observability/summary").json()
    assert body["refusals_by_reason"].get("no_results", 0) >= 1


def test_dashboard_serves_html(client):
    r = client.get("/observability/dashboard")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "observability/summary" in r.text  # the page actually fetches the summary endpoint


def test_response_request_id_header_matches_the_audit_log(client, caplog):
    import logging

    caplog.set_level(logging.INFO, logger="audit")
    r = client.get("/health")
    header_id = r.headers["X-Request-ID"]
    logged_ids = [rec.request_id for rec in caplog.records if hasattr(rec, "request_id")]
    assert header_id in logged_ids


def test_one_trace_id_unifies_audit_log_and_internal_service_logs(client, embedder, caplog):
    """The actual Day 9 promise: before today, the audit log's id and each
    router's internal service-log id (search/index/ask) were DIFFERENT
    values for the same request. Now they must be the SAME value."""
    import logging

    doc = client.post("/documents/upload", files={"file": ("a.pdf", _pdf(), "application/pdf")}).json()["document_id"]

    caplog.set_level(logging.INFO)
    caplog.clear()  # isolate to just the /index request below; upload already happened above
    r = client.post(f"/documents/{doc}/index")
    header_id = r.headers["X-Request-ID"]

    audit_records = [rec for rec in caplog.records if rec.name == "audit"]
    service_records = [rec for rec in caplog.records if rec.name.startswith("app.services") and hasattr(rec, "request_id")]

    assert audit_records and audit_records[0].request_id == header_id
    assert service_records, "expected at least one service-layer log line with a request_id"
    assert all(rec.request_id == header_id for rec in service_records), (
        "every service-layer log for this request must carry the SAME id as the audit log / response header"
    )
