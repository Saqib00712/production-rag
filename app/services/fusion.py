"""Reciprocal Rank Fusion (RRF).

BM25 scores (unbounded, corpus dependent) and cosine similarities ([-1,1])
live on incompatible scales, so we fuse RANKS, not scores:
    rrf(d) = sum over retrievers of 1 / (k + rank_r(d))
k=60 is the value from the original paper; it damps the influence of top
ranks so no single retriever dominates. Documents found by several
retrievers accumulate score and float to the top.
"""


def reciprocal_rank_fusion(
    rankings: dict[str, list[str]], k: int = 60
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for ids in rankings.values():
        for pos, chunk_id in enumerate(ids, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + pos)
            best_rank[chunk_id] = min(best_rank.get(chunk_id, pos), pos)
    # deterministic: score desc, then best individual rank, then id
    ordered = sorted(scores, key=lambda c: (-scores[c], best_rank[c], c))
    return [(c, scores[c]) for c in ordered]
