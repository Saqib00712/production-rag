"""
API key authentication.

Design decisions:
  * Bearer token in the Authorization header (`Authorization: Bearer sk-...`),
    the conventional place for an API credential -- not a query parameter
    (query params leak into logs, browser history, and proxy access logs;
    headers generally don't).
  * Keys are opaque random tokens, hashed with SHA-256 before storage and
    lookup (repositories/tenant_repository.py) -- the same principle as
    password storage. Losing the database does not leak usable keys.
  * A missing/invalid/revoked key returns 401 with no distinction between
    "wrong key" and "revoked key" in the response body -- distinguishing
    them would tell an attacker which keys once existed.
"""

import secrets
from dataclasses import dataclass

from fastapi import Depends, Header, HTTPException, Request, status

from app.config import Settings, get_settings
from app.dependencies import get_tenant_repository
from app.middleware.rate_limit import RateLimiter
from app.repositories.tenant_repository import TenantRepository

KEY_PREFIX = "sk-tenant-"


def generate_api_key() -> str:
    return KEY_PREFIX + secrets.token_hex(24)


@dataclass
class TenantContext:
    tenant_id: str
    key_label: str


def get_current_tenant(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    repo: TenantRepository = Depends(get_tenant_repository),
) -> TenantContext:
    """Accepts either `Authorization: Bearer <key>` or `X-API-Key: <key>`
    (some HTTP clients/tools make custom headers easier to set than Bearer
    auth) -- both resolve to the same lookup."""
    key = None
    if authorization and authorization.lower().startswith("bearer "):
        key = authorization[7:].strip()
    elif x_api_key:
        key = x_api_key.strip()

    if not key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide 'Authorization: Bearer <key>' or 'X-API-Key: <key>'.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    record = repo.lookup(key)
    if record is None or record.revoked:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or revoked API key.")
    return TenantContext(tenant_id=record.tenant_id, key_label=record.label)


def enforce_tenant_limits(
    request: Request,
    tenant: TenantContext = Depends(get_current_tenant),
    settings: Settings = Depends(get_settings),
    tenant_repo: TenantRepository = Depends(get_tenant_repository),
) -> TenantContext:
    """
    Second-layer, per-tenant guardrail on top of Day 7's per-IP middleware
    limiter: (1) a sliding-window rate limit keyed by tenant_id instead of
    IP (an authenticated identity is the right key once one exists -- Day 7
    flagged IP as a pre-auth proxy, not a permanent choice), and (2) a daily
    cost budget enforced BEFORE the request proceeds, not after.

    Implemented as a dependency (runs AFTER auth resolves tenant_id), not
    middleware -- middleware runs before route dependencies, so it cannot
    yet know which tenant is making the request.
    """
    limiter: "RateLimiter" = request.app.state.tenant_rate_limiter
    allowed, retry_after = limiter.allow(tenant.tenant_id)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Per-tenant rate limit exceeded. Please slow down and try again shortly.",
            headers={"Retry-After": str(int(retry_after) + 1)},
        )

    today = _today()
    spent_today = tenant_repo.get_usage(tenant.tenant_id, today)
    if spent_today >= settings.tenant_daily_budget_usd:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=f"Daily budget of ${settings.tenant_daily_budget_usd:.2f} reached for this tenant. Try again tomorrow.",
        )
    return tenant


def _today() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")


def record_tenant_spend(tenant_repo: TenantRepository, tenant_id: str, cost_usd: float) -> None:
    if cost_usd > 0:
        tenant_repo.add_usage(tenant_id, _today(), cost_usd)
