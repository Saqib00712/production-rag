"""
Benchmark: brute-force (exact) vs HNSW (approximate) vector search, at a
scale actually large enough for the difference to matter -- the whole
project's real documents so far have been a few dozen chunks, where Day 3's
brute-force scan is already faster than the network round-trip to embed a
query. This script answers "was upgrading to HNSW actually worth it" with
MEASURED numbers instead of assuming ANN is always better.

Uses synthetic random vectors, not real OpenAI embeddings: this benchmark
is testing the INDEX DATA STRUCTURE's speed/recall trade-off, which doesn't
depend on what the vectors mean semantically -- only on how many there are
and how expensive it is to scan or graph-search them. $0, no API key needed.

    python scripts/bench_vector_index.py                        # defaults: 20,000 chunks, 50 queries
    python scripts/bench_vector_index.py --n-chunks 100000 --n-queries 100
    python scripts/bench_vector_index.py --ef-search 20 100 200  # compare multiple ef_search settings
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-chunks", type=int, default=20_000)
    ap.add_argument("--n-queries", type=int, default=50)
    ap.add_argument("--dim", type=int, default=1536, help="matches text-embedding-3-small's real dimension")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--ef-search", type=int, nargs="+", default=[50])
    ap.add_argument("--ef-construction", type=int, default=200)
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-clusters", type=int, default=200,
                     help="synthetic 'topics' vectors are generated around -- see note below on why this matters")
    args = ap.parse_args()

    import numpy as np

    from app.services.vector_index import BruteForceIndex, HnswIndex

    rng = np.random.default_rng(args.seed)
    print(f"Generating {args.n_chunks:,} synthetic {args.dim}-dim vectors (clustered around "
          f"{args.n_clusters} topic centers) and {args.n_queries} queries...")
    # IMPORTANT: real embeddings are NOT uniform random noise -- semantically
    # related text clusters together in vector space, which is exactly the
    # structure HNSW's graph exploits to skip most of the index. Pure random
    # Gaussian vectors in high dimensions are close to the ADVERSARIAL worst
    # case for ANN (every point is almost equidistant from every other --
    # the "curse of dimensionality"), so benchmarking against uniform noise
    # would understate HNSW's real-world recall. We simulate realistic
    # structure instead: pick N cluster centers ("topics"), then generate
    # each vector as a center plus small noise ("a chunk about that topic") --
    # much closer to how real embeddings actually distribute.
    centers = rng.normal(size=(args.n_clusters, args.dim)).astype(np.float32)
    cluster_assignments = rng.integers(0, args.n_clusters, size=args.n_chunks)
    noise_scale = 0.3
    ids = [f"chunk-{i}" for i in range(args.n_chunks)]
    vectors = (centers[cluster_assignments] + rng.normal(scale=noise_scale, size=(args.n_chunks, args.dim))).astype(np.float32)
    query_clusters = rng.integers(0, args.n_clusters, size=args.n_queries)
    queries = (centers[query_clusters] + rng.normal(scale=noise_scale, size=(args.n_queries, args.dim))).astype(np.float32)

    print("\nBuilding brute-force index (ground truth)...")
    t0 = time.perf_counter()
    bf = BruteForceIndex()
    bf.build(ids, vectors)
    bf_build_s = time.perf_counter() - t0
    print(f"  build time: {bf_build_s:.2f}s")

    print("\nRunning brute-force queries...")
    t0 = time.perf_counter()
    bf_results = [bf.search(q.tolist(), args.k) for q in queries]
    bf_query_ms = (time.perf_counter() - t0) / args.n_queries * 1000
    print(f"  avg query latency: {bf_query_ms:.2f}ms")
    ground_truth = [{cid for cid, _ in r} for r in bf_results]

    rows = []
    for ef_search in args.ef_search:
        print(f"\nBuilding HNSW index (ef_construction={args.ef_construction}, M={args.m})...")
        t0 = time.perf_counter()
        hnsw = HnswIndex(ef_construction=args.ef_construction, m=args.m, ef_search=ef_search)
        hnsw.build(ids, vectors)
        hnsw_build_s = time.perf_counter() - t0
        print(f"  build time: {hnsw_build_s:.2f}s")

        print(f"Running HNSW queries (ef_search={ef_search})...")
        t0 = time.perf_counter()
        hnsw_results = [hnsw.search(q.tolist(), args.k) for q in queries]
        hnsw_query_ms = (time.perf_counter() - t0) / args.n_queries * 1000

        recalls = [
            len(ground_truth[i] & {cid for cid, _ in hnsw_results[i]}) / args.k
            for i in range(args.n_queries)
        ]
        avg_recall = sum(recalls) / len(recalls)
        speedup = bf_query_ms / hnsw_query_ms if hnsw_query_ms > 0 else float("inf")
        print(f"  avg query latency: {hnsw_query_ms:.2f}ms  (speedup: {speedup:.1f}x)")
        print(f"  avg recall@{args.k} vs. brute-force ground truth: {avg_recall:.2%}")

        rows.append({
            "ef_search": ef_search, "build_s": round(hnsw_build_s, 3),
            "query_ms": round(hnsw_query_ms, 3), "speedup_x": round(speedup, 2),
            "recall_at_k": round(avg_recall, 4),
        })

    print(f"\n{'='*70}")
    print(f"{'ef_search':<12} {'build(s)':<10} {'query(ms)':<11} {'speedup':<9} {'recall@'+str(args.k):<10}")
    print("-" * 70)
    print(f"{'brute-force':<12} {bf_build_s:<10.2f} {bf_query_ms:<11.2f} {'1.0x':<9} {'1.0000':<10} (ground truth)")
    for r in rows:
        print(f"{r['ef_search']:<12} {r['build_s']:<10.2f} {r['query_ms']:<11.2f} "
              f"{str(r['speedup_x'])+'x':<9} {r['recall_at_k']:<10}")

    print(f"\nInterpretation: recall@{args.k} close to 1.0 means HNSW is returning essentially the same results as "
          f"exact search. A speedup below ~1x at this scale means brute-force is still fine -- HNSW's advantage "
          f"grows with corpus size, so re-run with a larger --n-chunks if the speedup looks unimpressive here.")

    out_dir = Path("evals/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"vector-bench-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out_path.write_text(json.dumps({
        "n_chunks": args.n_chunks, "n_queries": args.n_queries, "dim": args.dim, "k": args.k,
        "brute_force": {"build_s": round(bf_build_s, 3), "query_ms": round(bf_query_ms, 3)},
        "hnsw_runs": rows,
    }, indent=2), encoding="utf-8")
    print(f"\nFull results written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
