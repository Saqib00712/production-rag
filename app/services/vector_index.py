"""
Pluggable vector index backends: exact (brute-force, Day 3) and approximate
(HNSW, Day 12) behind one shared interface, so SearchService's code never
needs to know which one is in use.

Why ANN at all: Day 3's brute-force search is a full matrix-vector product
over EVERY stored vector for EVERY query -- O(n) per query. At a few dozen
or even a few thousand chunks this is faster than the network round trip to
embed the query in the first place, which is why it was the right choice
through Day 11. Past tens of thousands of chunks, that O(n) scan starts to
dominate latency. HNSW (Hierarchical Navigable Small World) trades a small,
tunable amount of RECALL (it may occasionally miss the true best match) for
large latency wins at scale, by building a graph structure that lets search
skip most of the vectors instead of scanning all of them.

Why hnswlib specifically: a focused, widely-used (it's what Chroma and many
other vector stores embed internally), pip-installable C++ library with no
server to run -- consistent with this project's "a library, not a new piece
of infrastructure" bias (the same reasoning Day 9 applied to
prometheus_client over standing up a Prometheus server).

Why per-TENANT indexes, not one global index. Day 10 made tenant_id a
required filter on every chunk query specifically because a shared index
searched without that filter is a real data leak. hnswlib has no built-in
per-query metadata filtering, so the safe and simple design is one small
HNSW graph PER TENANT, built only from that tenant's vectors -- isolation
by construction, not by a filter that could be forgotten. The trade-off:
many small indexes instead of one large one, which is the right trade for
a system where tenant boundaries are a hard security requirement, not just
an optimization concern.
"""

import logging
from typing import Protocol

import numpy as np

from app.services.vector_math import cosine_top_k

logger = logging.getLogger(__name__)


class VectorIndex(Protocol):
    def build(self, ids: list[str], vectors: np.ndarray) -> None: ...
    def search(self, query: list[float], k: int) -> list[tuple[str, float]]: ...

    @property
    def size(self) -> int: ...


class BruteForceIndex:
    """Exact cosine search -- a thin wrapper around Day 3's cosine_top_k,
    kept as its own class so it satisfies the same VectorIndex interface as
    HnswIndex and both are swappable from SearchService without an if/else
    at the call site."""

    def __init__(self):
        self._ids: list[str] = []
        self._matrix: np.ndarray = np.empty((0, 0), dtype=np.float32)

    def build(self, ids: list[str], vectors: np.ndarray) -> None:
        self._ids = ids
        self._matrix = vectors

    def search(self, query: list[float], k: int) -> list[tuple[str, float]]:
        return cosine_top_k(query, self._ids, self._matrix, k)

    @property
    def size(self) -> int:
        return len(self._ids)


class HnswIndex:
    """
    Approximate cosine search via hnswlib.

    Tuning knobs (see config.py):
      ef_construction - how hard to work when BUILDING the graph (higher =
        better recall, slower build). Build happens once per index refresh,
        not per query, so this can be generous.
      m - max connections per node in the graph (higher = better recall,
        more memory). hnswlib's own default (16) is a reasonable start.
      ef_search - how hard to work when QUERYING (higher = better recall,
        slower search). This is the knob you'd actually tune live against
        a recall target, since it costs latency on every query.
    hnswlib computes inner product / L2 natively; cosine similarity over
    L2-normalized vectors is mathematically equivalent to using the 'cosine'
    space directly, which is what we request -- no manual normalization
    needed.
    """

    def __init__(self, ef_construction: int = 200, m: int = 16, ef_search: int = 50):
        self._ef_construction = ef_construction
        self._m = m
        self._ef_search = ef_search
        self._ids: list[str] = []
        self._index = None  # type: ignore[assignment]

    def build(self, ids: list[str], vectors: np.ndarray) -> None:
        try:
            import hnswlib
        except ImportError as exc:
            # hnswlib is deliberately NOT in requirements.txt / requirements-dev.txt
            # (Day 20) -- it's a compiled C++ extension that needs a Windows build
            # toolchain to build from source, which the default brute_force
            # backend has no reason to force on everyone. See requirements-hnsw.txt.
            raise RuntimeError(
                "vector_index_backend='hnsw' is selected but hnswlib isn't installed. "
                "Install it with:  pip install -r requirements-hnsw.txt "
                "(or switch back to the default vector_index_backend='brute_force')."
            ) from exc

        self._ids = ids
        if not ids:
            self._index = None
            return
        dim = vectors.shape[1]
        index = hnswlib.Index(space="cosine", dim=dim)
        index.init_index(max_elements=len(ids), ef_construction=self._ef_construction, M=self._m)
        index.add_items(vectors, np.arange(len(ids)))
        index.set_ef(max(self._ef_search, 1))
        self._index = index

    def search(self, query: list[float], k: int) -> list[tuple[str, float]]:
        if self._index is None or not self._ids:
            return []
        k = min(k, len(self._ids))
        q = np.asarray(query, dtype=np.float32).reshape(1, -1)
        try:
            labels, distances = self._index.knn_query(q, k=k)
        except RuntimeError as exc:
            # hnswlib raises if k exceeds the number of indexed elements in
            # some versions despite the clamp above (e.g. a race with a
            # concurrent rebuild) -- fail soft to no results rather than
            # crashing a search request over an index/tuning edge case.
            logger.warning("hnsw_query_failed", extra={"error_type": type(exc).__name__})
            return []
        # hnswlib's 'cosine' space returns DISTANCE (1 - cosine_similarity);
        # convert back to similarity so callers see the same scale as
        # BruteForceIndex / cosine_top_k.
        return [(self._ids[i], float(1.0 - d)) for i, d in zip(labels[0], distances[0])]

    @property
    def size(self) -> int:
        return len(self._ids)


def build_index(backend: str, ids: list[str], vectors: np.ndarray, *,
                 ef_construction: int = 200, m: int = 16, ef_search: int = 50) -> VectorIndex:
    if backend == "hnsw":
        index = HnswIndex(ef_construction=ef_construction, m=m, ef_search=ef_search)
    elif backend == "brute_force":
        index = BruteForceIndex()
    else:
        raise ValueError(f"Unknown vector index backend: {backend!r} (expected 'brute_force' or 'hnsw')")
    index.build(ids, vectors)
    return index
