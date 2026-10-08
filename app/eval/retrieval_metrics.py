"""
Pure retrieval-quality metrics. No I/O, no OpenAI -- these are unit-testable
with plain lists, which is why they live separately from the runner.

Definitions (the interview-ready version):
  hit@k    - did AT LEAST ONE expected page appear in the top k results?
             Binary per-question; averaged across a dataset it's a "success
             rate". Cheap to compute, easy to explain to non-engineers.
  MRR      - Mean Reciprocal Rank: 1 / (rank of the FIRST expected page
             found), 0 if never found. Rewards finding the answer EARLY,
             not just eventually -- a hit at rank 1 scores 1.0, at rank 10
             scores 0.1. Better than hit@k at distinguishing "great" from
             "barely adequate" retrieval.
  recall@k - of ALL expected pages for a question, what fraction appeared
             in the top k? Matters when an answer is spread across several
             pages; hit@k alone would call that a full success at one page.
"""


def compute_page_ranks(hit_pages: list[int], expected_pages: list[int]) -> dict[int, int]:
    """For each expected page, the EARLIEST 1-indexed rank it appeared at
    in hit_pages. Expected pages never found are simply absent from the
    result (not an error -- that's what a miss looks like)."""
    ranks: dict[int, int] = {}
    expected = set(expected_pages)
    for rank, page in enumerate(hit_pages, start=1):
        if page in expected and page not in ranks:
            ranks[page] = rank
    return ranks


def hit_at_k(page_ranks: dict[int, int], k: int) -> bool:
    return any(r <= k for r in page_ranks.values())


def reciprocal_rank(page_ranks: dict[int, int]) -> float:
    if not page_ranks:
        return 0.0
    return 1.0 / min(page_ranks.values())


def recall_at_k(page_ranks: dict[int, int], expected_pages: list[int], k: int) -> float:
    if not expected_pages:
        # An unanswerable question has no pages to recall. It is scored
        # separately (correct refusal vs false answer), not here -- 1.0
        # would misleadingly count as "perfect retrieval" for a query that
        # was never supposed to retrieve anything.
        return 0.0
    found = {p for p, r in page_ranks.items() if r <= k}
    return len(found) / len(set(expected_pages))
