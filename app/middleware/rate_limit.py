"""
In-memory sliding-window rate limiting middleware.

Design decisions:
  * Sliding window over fixed window: a fixed window (e.g. "20 requests per
    clock-minute") lets a client burst 40 requests across a window boundary
    (20 at 0:59, 20 more at 1:00). A sliding window (20 requests in any
    trailing 60 seconds) doesn't have that gap.
  * In-memory, per-process: correct and simple for local dev and a single
    server instance. WRONG the moment you run more than one worker/replica,
    since each process has its own counters -- a client could get up to
    N x limit through by hitting different workers. Flagged explicitly here
    and in docs/day7.md: the production fix is a shared store (Redis), which
    is exactly why "Redis" and "LLM Gateway with Rate Limiting" are later
    items on the curriculum, not a gap to quietly work around today.
  * Keyed by client IP by default. Once Day 10 (multi-tenant) adds API
    keys/auth, the key function should switch to the authenticated
    identity -- an IP is a poor proxy for "user" behind NAT or a shared
    office network, but it's the only identity we have pre-auth.
  * `time_fn` is injectable so tests control time exactly, without real
    sleeps -- the same testability pattern used throughout this project.
"""

import time
from collections import defaultdict, deque
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


class RateLimiter:
    """The counting logic, separated from the ASGI middleware plumbing so
    it's directly unit-testable with a fake clock and no HTTP involved."""

    def __init__(self, limit: int, window_seconds: float, time_fn: Callable[[], float] = time.monotonic):
        self.limit = limit
        self.window = window_seconds
        self._time_fn = time_fn
        self._hits: dict[str, deque] = defaultdict(deque)

    def allow(self, key: str) -> tuple[bool, float]:
        """Returns (allowed, retry_after_seconds). retry_after is 0 when allowed."""
        now = self._time_fn()
        q = self._hits[key]
        cutoff = now - self.window
        while q and q[0] <= cutoff:
            q.popleft()
        if len(q) >= self.limit:
            retry_after = round(q[0] + self.window - now, 2)
            return False, max(retry_after, 0.0)
        q.append(now)
        return True, 0.0

    def reset(self) -> None:
        self._hits.clear()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Applies a RateLimiter to a set of path prefixes (the expensive,
    paid-API-calling ones), leaving cheap endpoints like /health unlimited."""

    def __init__(self, app, limiter: RateLimiter, protected_prefixes: tuple[str, ...]):
        super().__init__(app)
        self._limiter = limiter
        self._prefixes = protected_prefixes

    async def dispatch(self, request: Request, call_next):
        if not any(request.url.path.startswith(p) for p in self._prefixes):
            return await call_next(request)

        key = _client_key(request)
        allowed, retry_after = self._limiter.allow(key)
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Please slow down and try again shortly."},
                headers={"Retry-After": str(int(retry_after) + 1)},
            )
        return await call_next(request)


def _client_key(request: Request) -> str:
    # A single trusted reverse proxy may set X-Forwarded-For; in a real
    # deployment this must come only from a proxy you control, never taken
    # as-is from the public internet, or a client can spoof any key/limit
    # bucket it likes by setting this header itself.
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"
