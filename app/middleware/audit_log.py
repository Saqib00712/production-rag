"""
Audit logging middleware: one structured log line per HTTP request, for
every route, with no per-router boilerplate.

Day 9 unifies what was, through Day 8, two separate uncorrelated ids for
one HTTP request: this middleware's own audit id, and each router's private
`str(uuid.uuid4())` used for its internal service-layer logging (search,
indexing, /ask). `tracing.start_trace()` sets ONE id that both this
middleware AND every router now read via `tracing.get_trace_id()` -- see
app/observability/tracing.py for why a contextvar is the right tool here.

This is also where the full per-request span breakdown (retrieval, rerank,
generation -- recorded via `tracing.span()` inside the pipeline) is logged
as ONE consolidated trace line, and where Prometheus HTTP metrics + the
local latency recorder (app/observability/metrics.py) are updated.
"""

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.observability import tracing
from app.observability.metrics import http_request_duration_seconds, http_requests_total, latency_recorder

logger = logging.getLogger("audit")
# Audit lines are the record of what happened on every request -- always
# emit them regardless of the app's general LOG_LEVEL (which may be raised
# to WARNING to quiet noisy third-party loggers; see logging_config.py).
# Without this, a quieted root logger silently drops INFO-level audit lines,
# which is exactly what made scripts/verify_day9.py's live check look like
# the audit middleware was broken when LOG_LEVEL=WARNING was set -- it
# wasn't; the line was just never emitted at that level.
logger.setLevel(logging.INFO)


class AuditLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        trace_id = str(uuid.uuid4())
        tracing.start_trace(trace_id)
        start = time.perf_counter()
        client_ip = request.client.host if request.client else "unknown"
        status_code = 500
        response: Response | None = None
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            duration_s = time.perf_counter() - start
            spans = tracing.get_spans()
            is_error = status_code >= 500
            latency_recorder.record(duration_s, is_error=is_error)
            http_requests_total.labels(method=request.method, path=request.url.path, status=str(status_code)).inc()
            http_request_duration_seconds.labels(method=request.method, path=request.url.path).observe(duration_s)

            log_fn = logger.error if is_error else logger.info
            log_fn(
                "http_request_failed" if is_error else "http_request",
                extra={
                    "request_id": trace_id, "method": request.method, "path": request.url.path,
                    "status_code": status_code, "client_ip": client_ip,
                    "duration_ms": round(duration_s * 1000, 2),
                    "spans": [{"name": s.name, "duration_ms": s.duration_ms, **s.attributes} for s in spans],
                },
            )
            if response is not None:
                response.headers["X-Request-ID"] = trace_id
            tracing.reset()
