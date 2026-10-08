from app.observability.metrics import CostTracker, LatencyRecorder, render_metrics


def test_latency_recorder_percentiles_on_known_data():
    r = LatencyRecorder()
    for ms in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        r.record(ms / 1000)
    assert r.percentile(50) in (50.0, 60.0)  # median of 10 sorted values, index rounding
    assert r.percentile(100) == 100.0
    assert r.percentile(0) == 10.0


def test_latency_recorder_empty_returns_zero():
    r = LatencyRecorder()
    assert r.percentile(50) == 0.0 and r.count == 0


def test_latency_recorder_tracks_error_rate():
    r = LatencyRecorder()
    r.record(0.01, is_error=False)
    r.record(0.02, is_error=True)
    r.record(0.03, is_error=False)
    assert r.total_requests == 3
    assert r.error_rate == round(1 / 3, 4)


def test_latency_recorder_capacity_evicts_oldest():
    r = LatencyRecorder(capacity=3)
    for ms in [10, 20, 30, 40]:
        r.record(ms / 1000)
    assert r.count == 3
    assert r.percentile(0) == 20.0  # the 10ms sample was evicted


def test_latency_recorder_reset_clears_everything():
    r = LatencyRecorder()
    r.record(0.5, is_error=True)
    r.reset()
    assert r.count == 0 and r.total_requests == 0 and r.error_rate == 0.0


def test_cost_tracker_accumulates_across_stages():
    c = CostTracker()
    c.record_cost("embedding", 100, 0.000002)
    c.record_cost("generation", 500, 0.0003)
    c.record_cost("embedding", 50, 0.000001)
    assert c.tokens_by_stage == {"embedding": 150, "generation": 500}
    assert abs(c.total_cost_usd - 0.000303) < 1e-9


def test_cost_tracker_tracks_cost_by_stage_separately_from_tokens():
    """Day 17: cost_by_stage is the dollar-denominated twin of
    tokens_by_stage -- same accumulation rule, but $ rather than tokens, so
    a cheap embedding stage and an expensive chat stage with similar token
    counts don't look equally costly."""
    c = CostTracker()
    c.record_cost("embedding", 100, 0.000002)
    c.record_cost("generation", 500, 0.0003)
    c.record_cost("embedding", 50, 0.000001)
    assert c.cost_by_stage == {"embedding": 0.000003, "generation": 0.0003}


def test_cost_tracker_cache_hit_ratio():
    c = CostTracker()
    for _ in range(3):
        c.record_cache("embedding", hit=True)
    c.record_cache("embedding", hit=False)
    assert c.cache_hit_ratio == 0.75


def test_cost_tracker_cache_ratio_with_no_events_is_zero():
    c = CostTracker()
    assert c.cache_hit_ratio == 0.0


def test_cost_tracker_refusal_counts():
    c = CostTracker()
    c.record_refusal("no_results")
    c.record_refusal("no_results")
    c.record_refusal("low_relevance")
    assert c.refusals_by_reason == {"no_results": 2, "low_relevance": 1}


def test_cost_tracker_reset():
    c = CostTracker()
    c.record_cost("x", 10, 0.01)
    c.record_cache("query", hit=True)
    c.record_refusal("no_results")
    c.reset()
    assert c.total_cost_usd == 0.0 and c.tokens_by_stage == {} and c.cache_hit_ratio == 0.0
    assert c.refusals_by_reason == {}
    assert c.cost_by_stage == {}


def test_render_metrics_produces_prometheus_text_format():
    body, content_type = render_metrics()
    assert b"http_requests_total" in body or b"# HELP" in body
    assert "text/plain" in content_type
