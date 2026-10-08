"""
Retrieval service: keyword (BM25), vector (cosine) and hybrid (RRF).

Production behaviours worth knowing:
  * Cheap-first: keyword search runs before any paid embedding call.
  * Graceful degradation: if the embedding provider is down, HYBRID returns
    keyword-only results flagged degraded=True instead of failing.
    (mode="vector" has no fallback, so it fails loudly.)
  * Query embeddings are cached (repeat queries cost nothing).
  * User input never reaches FTS5 raw (see build_match_query).
"""

import hashlib
import logging
import re
import time
from typing import Callable

from starlette.concurrency import run_in_threadpool

from app.config import Settings
from app.models.search import SearchHit, SearchMode, SearchResponse
from app.repositories.chunk_repository import ChunkRepository
from app.services.cost import estimate_cost_usd
from app.services.embedder import Embedder, EmbeddingConfigError, EmbeddingError
from app.services.fusion import reciprocal_rank_fusion
from app.services.vector_index_cache import TenantIndexCache
from app.services.vector_math import cosine_top_k

logger = logging.getLogger(__name__)

_STOPWORDS = frozenset(
    "a an and are as at be by for from how i in is it of on or that the this to was "
    "what when where which who why with do does did can could should would my me you "
    "your we our their there".split()
)


class SearchError(Exception):
    """Search cannot be answered consistently (e.g. embedding dimension mismatch)."""


def build_match_query(query: str, max_terms: int = 32) -> str:
    """User text -> safe FTS5 query: '"term1" OR "term2"'.

    Only \\w+ tokens survive and each is quoted, so FTS5 operators
    (AND/OR/NOT/NEAR, quotes, colons, parentheses, '-') in user input are
    inert. OR semantics + BM25 means partial matches still rank, with rarer
    terms weighing more (IDF).
    """
    terms: list[str] = []
    seen: set[str] = set()
    for tok in re.findall(r"\w+", query.lower()):
        if tok in _STOPWORDS or tok in seen or (len(tok) < 2 and not tok.isdigit()):
            continue
        seen.add(tok)
        terms.append(tok)
        if len(terms) >= max_terms:
            break
    return " OR ".join(f'"{t}"' for t in terms)


class SearchService:
    def __init__(
        self,
        repo: ChunkRepository,
        embedder_factory: Callable[[], Embedder],
        settings: Settings,
        index_cache: TenantIndexCache | None = None,
    ):
        self._repo = repo
        self._embedder_factory = embedder_factory
        self._settings = settings
        # Optional: when provided, whole-tenant vector searches (no
        # document_id filter) use a cached ANN/brute-force index instead of
        # loading and scanning every vector on every query -- see
        # services/vector_index_cache.py and docs/day12.md. A single-
        # document search stays on the direct load_vectors() path below
        # regardless: that result set is already small, so a per-document
        # cache would add complexity without a measurable benefit.
        self._index_cache = index_cache

    async def search(
        self,
        *,
        query: str,
        mode: SearchMode,
        top_k: int,
        document_id: str | None,
        tenant_id: str,
        request_id: str,
    ) -> SearchResponse:
        t_start = time.perf_counter()
        timings: dict[str, float] = {}
        warnings: list[str] = []
        degraded = False
        depth = max(top_k * self._settings.search_candidate_multiplier, 20)
        kw: list[tuple[str, float]] = []
        vec: list[tuple[str, float]] = []
        q_tokens = 0

        if mode in ("keyword", "hybrid"):
            t = time.perf_counter()
            match = build_match_query(query)
            kw = await run_in_threadpool(self._repo.keyword_search, match, depth, tenant_id, document_id)
            timings["keyword"] = _ms(t)
            if not match:
                warnings.append("Query has no searchable terms after removing stopwords.")

        if mode in ("vector", "hybrid"):
            try:
                vec, q_tokens = await self._vector(query, depth, document_id, tenant_id, timings)
            except (EmbeddingError, EmbeddingConfigError) as exc:
                if mode == "vector":
                    raise
                degraded = True
                warnings.append(f"Vector retrieval unavailable ({type(exc).__name__}); returned keyword-only results.")
                logger.warning("hybrid_degraded", extra={"request_id": request_id, "error_type": type(exc).__name__})

        # ---- final ranking ----
        if mode == "keyword" or (mode == "hybrid" and degraded):
            score_type, ranked = "bm25", kw
        elif mode == "vector":
            score_type, ranked = "cosine", vec
        else:
            score_type = "rrf"
            ranked = reciprocal_rank_fusion(
                {"keyword": [c for c, _ in kw], "vector": [c for c, _ in vec]},
                k=self._settings.rrf_k,
            )
        ranked = ranked[:top_k]

        kw_rank = {c: i for i, (c, _) in enumerate(kw, 1)}
        kw_score = dict(kw)
        vec_rank = {c: i for i, (c, _) in enumerate(vec, 1)}
        vec_score = dict(vec)
        chunks = await run_in_threadpool(self._repo.get_chunks_by_ids, [c for c, _ in ranked], tenant_id)

        hits = [
            SearchHit(
                rank=i, chunk_id=cid, document_id=chunks[cid].document_id,
                page_number=chunks[cid].page_number, text=chunks[cid].text, score=score,
                keyword_rank=kw_rank.get(cid), keyword_score=kw_score.get(cid),
                vector_rank=vec_rank.get(cid), vector_score=vec_score.get(cid),
            )
            for i, (cid, score) in enumerate(ranked, 1)
            if cid in chunks
        ]
        timings["total"] = _ms(t_start)
        cost = estimate_cost_usd(q_tokens, self._settings.embedding_price_per_million_tokens)
        logger.info(
            "search",
            extra={"request_id": request_id, "mode": mode, "hits": len(hits),
                   "duration_ms": timings["total"], "tokens": q_tokens, "cost_usd": cost},
        )
        return SearchResponse(
            query=query, mode=mode, score_type=score_type, top_k=top_k,
            document_id=document_id, hits=hits, degraded=degraded, warnings=warnings,
            query_tokens=q_tokens, query_cost_usd=cost, timings_ms=timings,
        )

    async def _vector(self, query, depth, document_id, tenant_id, timings):
        t = time.perf_counter()
        qvec, tokens = await self._embed_query(query)
        timings["embed_query"] = _ms(t)
        t = time.perf_counter()
        model = self._settings.openai_embedding_model

        if document_id is None and self._index_cache is not None:
            # Whole-tenant search: use the cached per-tenant index (Day 12)
            # instead of loading + scanning every vector on every query.
            async def loader():
                return await run_in_threadpool(self._repo.load_vectors, model, tenant_id, None)

            index = await self._index_cache.get(tenant_id, model, loader)
            known_dim = _probe_dim(index)
            if index.size and known_dim >= 0 and len(qvec) != known_dim:
                raise SearchError(
                    "Stored vectors have a different dimension than the query embedding; re-index your documents."
                )
            results = index.search(qvec, depth)
        else:
            ids, matrix = await run_in_threadpool(self._repo.load_vectors, model, tenant_id, document_id)
            if ids and matrix.shape[1] != len(qvec):
                raise SearchError(
                    f"Stored vectors have {matrix.shape[1]} dims but the query has {len(qvec)}; re-index the document."
                )
            results = cosine_top_k(qvec, ids, matrix, depth)
        timings["vector"] = _ms(t)
        return results, tokens

    async def _embed_query(self, query: str) -> tuple[list[float], int]:
        model = self._settings.openai_embedding_model
        qhash = hashlib.sha256(query.encode("utf-8")).hexdigest()
        cached = await run_in_threadpool(self._repo.get_query_embedding, qhash, model)
        if cached is not None:
            return cached, 0
        result = await self._embedder_factory().embed([query])
        if len(result.vectors) != 1:
            raise EmbeddingError("Provider returned an unexpected number of vectors.")
        await run_in_threadpool(
            self._repo.save_query_embedding, qhash, model, result.vectors[0],
            self._settings.query_cache_max_rows,
        )
        return result.vectors[0], result.total_tokens


def _ms(since: float) -> float:
    return round((time.perf_counter() - since) * 1000, 2)


def _probe_dim(index) -> int:
    """Best-effort dimension check across index backends without requiring
    VectorIndex to expose a dedicated property just for this one guard."""
    matrix = getattr(index, "_matrix", None)
    if matrix is not None and getattr(matrix, "ndim", 0) == 2 and matrix.shape[0] > 0:
        return matrix.shape[1]
    return -1  # unknown (e.g. HnswIndex): skip the dimension check rather than guess wrong
