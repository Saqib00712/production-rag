"""
Application entrypoint.

Design decision: main.py wires things together (logging, routers) and
contains no business logic itself. This keeps the app importable and
testable (tests import `app` without triggering side effects beyond what's
explicitly configured here).
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import get_settings
from app.logging_config import configure_logging
from app.middleware.audit_log import AuditLogMiddleware
from app.middleware.rate_limit import RateLimiter, RateLimitMiddleware
from app.auth import generate_api_key
from app.dependencies import get_tenant_repository
from app.observability.otel_export import init_otel
from app.routers import ask, conversations, documents, memories, observability, search
from app.services.openai_client import close_openai_clients

configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    init_otel(settings)  # Day 21: no-op unless otel_enabled=true and the SDK is installed
    if settings.auto_bootstrap_dev_key:
        tenant_repo = get_tenant_repository(settings)
        if not tenant_repo.any_keys_exist():
            key = generate_api_key()
            tenant_repo.create_key(key, tenant_id="default", label="auto-bootstrapped dev key")
            print("\n" + "=" * 70)
            print("No API keys existed yet -- created one for local development:")
            print(f"  {key}")
            print("Use it as:  Authorization: Bearer " + key)
            print("This is printed ONCE. Manage keys with scripts/manage_keys.py.")
            print("=" * 70 + "\n")
    yield
    # Closes the shared AsyncOpenAI client cleanly on shutdown, avoiding the
    # "Event loop is closed" warnings that per-request clients caused.
    await close_openai_clients()


app = FastAPI(
    title="Production RAG with Citations",
    version="0.21.0",
    description=(
        "Day 21: optional real OpenTelemetry export, PII redaction mode, and vision-based OCR for scanned PDFs"
    ),
    lifespan=lifespan,
)

app.include_router(documents.router)
app.include_router(search.router)
app.include_router(ask.router)
app.include_router(observability.router)
app.include_router(conversations.router)
app.include_router(memories.router)

_settings = get_settings()
_rate_limiter = RateLimiter(limit=_settings.rate_limit_per_minute, window_seconds=_settings.rate_limit_window_seconds)
app.state.rate_limiter = _rate_limiter  # exposed for tests to adjust/reset; see tests/test_middleware.py
app.add_middleware(RateLimitMiddleware, limiter=_rate_limiter, protected_prefixes=("/ask", "/documents"))
app.add_middleware(AuditLogMiddleware)

# Day 10: a SECOND limiter, keyed by authenticated tenant_id rather than IP
# (see app.auth.enforce_tenant_limits). Separate instance from the Day 7
# per-IP limiter above -- different key space, different purpose.
app.state.tenant_rate_limiter = RateLimiter(
    limit=_settings.tenant_rate_limit_per_minute, window_seconds=_settings.rate_limit_window_seconds
)


@app.get("/health")
def health() -> dict:
    """
    Liveness check. Deliberately has zero dependencies (no DB, no OpenAI
    call) so it answers instantly and can't false-negative because a
    downstream dependency is slow -- that's what a separate /readiness
    check would be for, once we have real dependencies to check (Day 2+).
    """
    return {"status": "ok"}
