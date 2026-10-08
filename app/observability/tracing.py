"""
Request-scoped trace context: ONE id per incoming request, propagated to
every layer (middleware, routers, services) without threading it through
every function signature by hand, plus a lightweight span recorder that
answers "where did the time go?" for a single request.

Why contextvars: this is the exact primitive real distributed-tracing
libraries (including OpenTelemetry itself) build on. A contextvar is
per-async-task state that flows automatically through `await` calls in the
SAME task, without global mutable state and without manual parameter
passing -- exactly what's needed here, since a request's work is one
async task from the middleware down through routers and services.

Why this module still exists now that real OTel export is available (Day
21, app/observability/otel_export.py): OTel's API surface (spans,
exporters, samplers, propagators, resource attributes) is built for
shipping traces to a real backend (Jaeger, Tempo, an OTel Collector) --
valuable, but a real piece of infrastructure to run, not a pip install away
from working locally, and most users of this project never need it. This
module still answers today's problem (one process, no collector, "why did
this request take 5.65s") with zero infrastructure and zero setup; `span()`
ADDITIONALLY opens a real OTel span when `settings.otel_enabled=true` and
the SDK is installed (otherwise that's a true no-op) -- one call site,
two optional outputs, exactly the upgrade path this docstring used to just
flag as future work.

Before Day 9, each router generated its OWN `str(uuid.uuid4())` per call,
and AuditLogMiddleware generated a SEPARATE id for the same HTTP request --
two different, uncorrelated ids for one request, which made "find every log
line for this one request" impossible without grepping two unrelated
values. `trace_id()` fixes that: one id, set once by the middleware, read
everywhere else.
"""

import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Iterator
from contextlib import contextmanager

from app.observability.otel_export import otel_span

_trace_id: ContextVar[str | None] = ContextVar("_trace_id", default=None)
_spans: ContextVar[list["Span"] | None] = ContextVar("_spans", default=None)


@dataclass
class Span:
    name: str
    start: float
    duration_ms: float | None = None
    attributes: dict = field(default_factory=dict)


def start_trace(trace_id: str) -> None:
    """Called once, by the middleware, at the start of a request."""
    _trace_id.set(trace_id)
    _spans.set([])


def get_trace_id(fallback_factory=None) -> str:
    """Read the current request's trace id. Outside a request (a script,
    a test with no middleware) there is none -- fallback_factory (e.g.
    `lambda: str(uuid.uuid4())`) generates a standalone id instead of
    raising, since not every caller runs inside an HTTP request."""
    tid = _trace_id.get()
    if tid is not None:
        return tid
    if fallback_factory is not None:
        return fallback_factory()
    import uuid

    return str(uuid.uuid4())


@contextmanager
def span(name: str, **attributes) -> Iterator[Span]:
    """Times a block of work and records it against the current trace.
    Usable with or without an active trace (spans are simply discarded if
    start_trace() was never called -- e.g. in a unit test that doesn't care
    about tracing at all)."""
    s = Span(name=name, start=time.perf_counter(), attributes=attributes)
    with otel_span(name, **attributes):  # Day 21: no-op unless otel_enabled + the SDK is installed
        try:
            yield s
        finally:
            s.duration_ms = round((time.perf_counter() - s.start) * 1000, 2)
            bucket = _spans.get()
            if bucket is not None:
                bucket.append(s)


def get_spans() -> list[Span]:
    return list(_spans.get() or [])


def reset() -> None:
    """Test/script helper: clears trace state so runs don't leak into each other."""
    _trace_id.set(None)
    _spans.set(None)
