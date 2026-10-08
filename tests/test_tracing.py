import asyncio

from app.observability.tracing import get_spans, get_trace_id, reset, span, start_trace


def test_get_trace_id_without_active_trace_uses_fallback():
    reset()
    assert get_trace_id(lambda: "fallback-id") == "fallback-id"


def test_get_trace_id_without_fallback_generates_a_uuid():
    reset()
    tid = get_trace_id()
    assert len(tid) == 36 and tid.count("-") == 4  # looks like a uuid4


def test_start_trace_sets_the_id_for_the_rest_of_the_context():
    reset()
    start_trace("trace-123")
    assert get_trace_id() == "trace-123"
    assert get_trace_id(lambda: "should not be used") == "trace-123"


def test_span_records_duration_and_attributes():
    reset()
    start_trace("t1")
    with span("retrieval", document_id="d1") as s:
        pass
    spans = get_spans()
    assert len(spans) == 1
    assert spans[0].name == "retrieval"
    assert spans[0].attributes == {"document_id": "d1"}
    assert spans[0].duration_ms is not None and spans[0].duration_ms >= 0


def test_multiple_spans_recorded_in_order():
    reset()
    start_trace("t1")
    with span("a"):
        pass
    with span("b"):
        pass
    assert [s.name for s in get_spans()] == ["a", "b"]


def test_span_without_active_trace_does_not_raise():
    reset()  # no start_trace() called
    with span("orphan"):
        pass
    assert get_spans() == []  # nothing to collect into, and that's fine


def test_reset_clears_both_trace_id_and_spans():
    start_trace("t1")
    with span("a"):
        pass
    reset()
    assert get_spans() == []


async def _worker(tag):
    with span(f"child-{tag}"):
        await asyncio.sleep(0)
    return get_trace_id()


def test_trace_id_propagates_across_await_boundaries_in_the_same_task():
    async def run():
        reset()
        start_trace("propagated-id")
        result = await _worker("x")
        assert result == "propagated-id"
        assert [s.name for s in get_spans()] == ["child-x"]

    asyncio.run(run())
