"""Exact (brute-force) cosine similarity search with numpy.

Why brute force: exact results (a perfect-recall baseline for evals), no
native extension to install, and fast enough for tens of thousands of
chunks (one matrix-vector product). Approximate indexes (HNSW/IVF) are
introduced when scale demands them (vector DB at scale project).
"""

import numpy as np


def cosine_top_k(
    query: list[float], ids: list[str], matrix: np.ndarray, k: int
) -> list[tuple[str, float]]:
    if not ids or k <= 0:
        return []
    q = np.asarray(query, dtype=np.float32)
    qn = float(np.linalg.norm(q))
    if qn == 0.0:
        return []
    norms = np.linalg.norm(matrix, axis=1)
    norms[norms == 0.0] = 1.0  # zero vectors score 0 instead of NaN
    scores = (matrix @ q) / (norms * qn)
    k = min(k, len(ids))
    top = np.argpartition(-scores, k - 1)[:k]
    top = top[np.argsort(-scores[top], kind="stable")]
    return [(ids[i], float(scores[i])) for i in top]
