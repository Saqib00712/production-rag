"""One shared AsyncOpenAI client per process.

Creating a client per request means a new connection pool + TLS handshake for
every call (slow) and un-awaited cleanup on shutdown (the 'Event loop is
closed' noise). We build it once and close it in the app lifespan.
"""

from app.config import Settings

_CLIENTS: dict[str, object] = {}


class OpenAIConfigError(Exception):
    """OPENAI_API_KEY missing or still the placeholder."""


def get_openai_client(settings: Settings):
    key = settings.openai_api_key
    if not key or key.startswith("sk-your"):
        raise OpenAIConfigError("OPENAI_API_KEY is not configured.")
    if key not in _CLIENTS:
        from openai import AsyncOpenAI

        _CLIENTS[key] = AsyncOpenAI(
            api_key=key, timeout=settings.llm_timeout_seconds, max_retries=settings.llm_max_retries
        )
    return _CLIENTS[key]


async def close_openai_clients() -> None:
    clients = list(_CLIENTS.values())
    _CLIENTS.clear()
    for c in clients:
        await c.close()
