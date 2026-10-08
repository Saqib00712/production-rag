"""
Per-tenant, in-process cache of built VectorIndex objects.

Building an HNSW graph (or even just assembling the brute-force matrix) has
a real cost -- doing it on every single query would throw away the whole
point of an ANN index. This cache builds an index ONCE per tenant and
reuses it across queries, rebuilding only when that tenant's data has
changed (`invalidate()`, called after indexing new/changed documents).

Concurrency note: this is a simple dict-based cache, not lock-protected
against two concurrent requests both deciding to rebuild the same tenant's
index at once. In the worst case that means redundant work, never incorrect
results (each rebuild reads a consistent snapshot from SQLite) -- an
acceptable trade for this project's scale, flagged here rather than
silently assumed safe at higher concurrency (a per-tenant asyncio.Lock
would be the production fix, see docs/day12.md).
"""

from typing import Awaitable, Callable

import numpy as np

from app.services.vector_index import VectorIndex, build_index

Loader = Callable[[], Awaitable[tuple[list[str], np.ndarray]]]


class TenantIndexCache:
    def __init__(self, backend: str, ef_construction: int, m: int, ef_search: int):
        self._backend = backend
        self._ef_construction = ef_construction
        self._m = m
        self._ef_search = ef_search
        self._indexes: dict[tuple[str, str], VectorIndex] = {}  # (tenant_id, model) -> index
        self._dirty: set[str] = set()  # tenant_ids needing a rebuild on next access

    async def get(self, tenant_id: str, model: str, loader: Loader) -> VectorIndex:
        key = (tenant_id, model)
        if key not in self._indexes or tenant_id in self._dirty:
            ids, matrix = await loader()
            self._indexes[key] = build_index(
                self._backend, ids, matrix,
                ef_construction=self._ef_construction, m=self._m, ef_search=self._ef_search,
            )
            self._dirty.discard(tenant_id)
        return self._indexes[key]

    def invalidate(self, tenant_id: str) -> None:
        self._dirty.add(tenant_id)

    def reset(self) -> None:
        self._indexes.clear()
        self._dirty.clear()
