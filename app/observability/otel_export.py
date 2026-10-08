"""
Optional real OpenTelemetry export (Day 21) -- the upgrade path flagged
explicitly in tracing.py's Day 9 docstring, now built.

Why this stays a SEPARATE module instead of rewriting tracing.py: Day 9's
lightweight `span()`/`Span` recorder (contextvars, zero infrastructure)
still answers "where did THIS request's time go" for the dashboard and
tests with no setup at all, and every existing caller/test depends on its
exact shape (a `Span` dataclass, `get_spans()`). This module does not
replace that -- it ADDS a second, optional side effect: when enabled, the
exact same `span(name, **attrs)` call ALSO opens a real OTel span, so one
call site produces both the free, zero-infra local view AND, if you choose
to turn it on, a real trace shippable to Jaeger/Tempo/any OTel collector.

Why fail-open, exactly like hnswlib (Day 20) and PyMuPDF OCR (Day 21): the
`opentelemetry-*` packages are a real dependency tree that most users of
this project will never need (otel_enabled defaults to False) -- they live
in requirements-otel.txt, not the hard requirements files. If they aren't
installed, or otel_enabled is False, `otel_span()` is a true no-op; nothing
about request handling changes, and no exception can ever propagate from
tracing into application code over an observability feature being off.
"""

import logging
from contextlib import contextmanager
from typing import Iterator

logger = logging.getLogger(__name__)

_tracer = None  # None = disabled or unavailable; set by init_otel()


def init_otel(settings) -> None:
    """Call once at startup (app/main.py's lifespan). Idempotent and safe
    to call even when otel_enabled is False or the SDK isn't installed --
    both cases just leave `_tracer` as None."""
    global _tracer
    _tracer = None
    if not settings.otel_enabled:
        return

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    except ImportError:
        logger.warning(
            "otel_enabled_but_sdk_not_installed",
            extra={"hint": "pip install -r requirements-otel.txt"},
        )
        return

    exporter = ConsoleSpanExporter()
    if settings.otel_exporter_otlp_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            exporter = OTLPSpanExporter(endpoint=settings.otel_exporter_otlp_endpoint)
        except ImportError:
            logger.warning(
                "otel_otlp_exporter_not_installed_falling_back_to_console",
                extra={"hint": "pip install -r requirements-otel.txt"},
            )

    provider = TracerProvider(resource=Resource.create({"service.name": settings.otel_service_name}))
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(settings.otel_service_name)
    logger.info("otel_initialized", extra={
        "exporter": "otlp" if settings.otel_exporter_otlp_endpoint else "console",
    })


def is_enabled() -> bool:
    return _tracer is not None


def reset_for_tests() -> None:
    """Test helper: forces OTel back to the disabled state regardless of
    what a previous init_otel() call did, so tests don't leak a real
    TracerProvider into unrelated test cases."""
    global _tracer
    _tracer = None


@contextmanager
def otel_span(name: str, **attributes) -> Iterator[None]:
    """A true no-op when OTel isn't enabled/installed -- see module
    docstring. attributes are filtered to OTel-safe primitive types since
    callers (tracing.span()) pass arbitrary **kwargs from application code
    that were never validated against OTel's stricter attribute typing."""
    if _tracer is None:
        yield
        return
    safe_attrs = {k: v for k, v in attributes.items() if isinstance(v, (str, bool, int, float))}
    with _tracer.start_as_current_span(name, attributes=safe_attrs):
        yield
