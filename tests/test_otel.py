"""
Unit tests for Day 21's optional OTel export
(app/observability/otel_export.py + its wiring into tracing.span()).

Covers three states: disabled (the default -- true no-op), enabled with the
SDK installed (a real span is recorded, using OTel's own InMemorySpanExporter
so nothing leaves the process), and enabled but the SDK NOT installed
(simulated via monkeypatching the import -- fails open, never raises).
"""

import pytest

from app.config import Settings
from app.observability import otel_export
from app.observability.tracing import span


@pytest.fixture(autouse=True)
def _reset_otel():
    """otel_export._tracer is a module-level singleton, same rationale as
    every other observability reset in conftest.py -- never let one test's
    TracerProvider leak into another."""
    otel_export.reset_for_tests()
    yield
    otel_export.reset_for_tests()


def test_disabled_by_default_is_a_true_noop():
    settings = Settings(otel_enabled=False)
    otel_export.init_otel(settings)
    assert otel_export.is_enabled() is False

    with span("some_stage", chunks=3) as s:
        pass
    assert s.duration_ms is not None  # the Day 9 local recorder still works, untouched


def test_enabled_but_sdk_not_installed_fails_open(monkeypatch):
    """Simulates a machine without `pip install -r requirements-otel.txt`."""
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "opentelemetry":
            raise ImportError("simulated: opentelemetry not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    settings = Settings(otel_enabled=True)
    otel_export.init_otel(settings)  # must not raise
    assert otel_export.is_enabled() is False

    with span("some_stage") as s:  # must not raise either
        pass
    assert s.duration_ms is not None


def test_init_otel_enables_tracing_end_to_end(capsys):
    """The real init_otel() path main.py's lifespan calls -- console
    exporter (otel_exporter_otlp_endpoint left empty), no network, safe to
    run in CI. Confirms the wiring works, not just the bypass the other
    tests use for isolation."""
    pytest.importorskip("opentelemetry")
    settings = Settings(otel_enabled=True, otel_service_name="test-service")
    otel_export.init_otel(settings)
    assert otel_export.is_enabled() is True

    with span("embedding", chunks=2):
        pass

    # ConsoleSpanExporter's BatchSpanProcessor flushes on its own schedule,
    # not synchronously -- force it so the assertion below is deterministic
    # rather than racing a background thread.
    from opentelemetry import trace

    trace.get_tracer_provider().force_flush()
    printed = capsys.readouterr().out
    assert '"name": "embedding"' in printed


def test_enabled_with_sdk_installed_records_a_real_span():
    otel = pytest.importorskip("opentelemetry")
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    # Build our own in-memory-exporting provider directly (bypassing
    # init_otel's own exporter choice) so this test captures the span
    # WITHOUT printing to the console or needing a real collector. Pulling
    # the tracer straight off THIS provider instance (not registering it
    # globally via trace.set_tracer_provider, which OTel only allows ONCE
    # per process) keeps each test's provider isolated from the others.
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    otel_export._tracer = provider.get_tracer("test")

    with span("embedding", chunks=5):
        pass

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "embedding"
    assert spans[0].attributes["chunks"] == 5


def test_non_primitive_attributes_are_filtered_not_crashed():
    """span() is called all over the app with arbitrary **kwargs that were
    never validated against OTel's stricter attribute typing (OTel only
    accepts str/bool/int/float and sequences thereof) -- a dict or None
    value must be dropped, not raise."""
    otel = pytest.importorskip("opentelemetry")
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    otel_export._tracer = provider.get_tracer("test")

    with span("weird_stage", count=3, weird=None, nested={"a": 1}):
        pass

    spans = exporter.get_finished_spans()
    assert spans[0].attributes["count"] == 3
    assert "weird" not in spans[0].attributes
    assert "nested" not in spans[0].attributes
