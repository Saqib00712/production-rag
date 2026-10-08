"""
One-command Day 3 verification.

    python scripts/verify_day3.py            # full check (uses your OpenAI key, costs < $0.001)
    python scripts/verify_day3.py --offline  # no OpenAI calls, keyword search only ($0)
    python scripts/verify_day3.py --skip-pytest

What it does
  1. Environment checks (Python, FTS5, numpy, tiktoken, API key present)
  2. Runs the whole pytest suite (no OpenAI calls)
  3. End-to-end run in a TEMP folder (your real data is not touched):
     build an 8-page PDF -> upload -> dry-run -> index -> re-index (cache)
     -> run 8 questions through keyword / vector / hybrid and print a table
Exit code 0 = everything passed, 1 = something failed.
"""

import argparse
import os
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ.setdefault("LOG_LEVEL", "WARNING")  # keep the output readable

RESULTS: list[tuple[bool, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((ok, name, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return ok


PAGES = [
    "Refund policy. Customers may request a refund within 30 days of purchase. Refunds are returned to the buyer within five working days.",
    "Warranty terms. The device carries a two year limited warranty covering manufacturing defects. Warranty claims require the serial number ERR-4471 printed on the box.",
    "Shipping information. Standard shipping takes five to seven business days. Express delivery arrives within two days for an extra fee.",
    "Privacy notice. We never sell personal data. Users may request deletion of their account information at any time.",
    "Payment options. We accept Visa, Mastercard and PayPal. Bank transfers are available for orders above five hundred dollars.",
    "Account help. To reset a forgotten password, click the reset link on the sign in page and follow the emailed instructions.",
    "Product specifications. The rechargeable battery provides up to twelve hours of continuous use. Charging takes two hours.",
    "Customer support. Our support team is available Monday to Friday from nine to five by telephone and live chat.",
]
# (question, expected page, what it tests)
QUESTIONS = [
    ("serial number ERR-4471", 2, "exact identifier"),
    ("how long do I have to send the item back to get my money?", 1, "paraphrase"),
    ("how fast will my order arrive", 3, "paraphrase"),
    ("can I have my personal information erased", 4, "paraphrase"),
    ("what payment methods do you accept", 5, "keyword+meaning"),
    ("I forgot my login credentials", 6, "paraphrase"),
    ("how long does the battery last", 7, "keyword+meaning"),
    ("when can I phone customer service", 8, "paraphrase"),
]


def build_pdf(pages: list[str]) -> bytes:
    from fpdf import FPDF

    pdf = FPDF()
    for text in pages:
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        pdf.multi_cell(0, 10, text)
    return bytes(pdf.output())


def step_environment(offline: bool) -> bool:
    print("\n[1/3] Environment")
    check("Python >= 3.10", sys.version_info >= (3, 10), sys.version.split()[0])
    try:
        c = sqlite3.connect(":memory:")
        c.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
        check("SQLite FTS5 available (keyword search)", True)
    except sqlite3.OperationalError:
        check("SQLite FTS5 available (keyword search)", False, "your Python's SQLite lacks FTS5")
    try:
        import numpy

        check("numpy installed", True, numpy.__version__)
    except ImportError:
        check("numpy installed", False, "pip install -r requirements.txt")
    from app.services.chunker import TokenCounter

    exact = TokenCounter().is_exact
    print(f"  [{'INFO' if exact else 'WARN'}] tiktoken {'loaded (exact token counts)' if exact else 'unavailable: token counts are approximate (not a failure)'}")
    from app.config import Settings

    key = Settings().openai_api_key
    has_key = bool(key) and not key.startswith("sk-your")
    if not offline:
        check("OPENAI_API_KEY found in .env", has_key, "" if has_key else "set it, or run with --offline")
    return has_key


def step_pytest() -> None:
    print("\n[2/3] Test suite (pytest)")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
        capture_output=True, text=True,
    )
    tail = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()][-1:] or ["(no output)"]
    check("pytest passes", proc.returncode == 0, tail[0])
    if proc.returncode != 0:
        print(proc.stdout[-3000:])
        print(proc.stderr[-1500:])


def rank_of(hits: list[dict], page: int) -> int | None:
    for h in hits:
        if h["page_number"] == page:
            return h["rank"]
    return None


def step_end_to_end(offline: bool) -> None:
    print("\n[3/3] End-to-end retrieval run (temp storage, your data is untouched)")
    from fastapi.testclient import TestClient

    from app.config import Settings, get_settings
    from app.dependencies import get_embedder_factory
    from app.main import app

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        settings = Settings(raw_storage_dir=f"{tmp}/raw", sqlite_path=f"{tmp}/verify.db")
        app.dependency_overrides[get_settings] = lambda: settings
        if offline:
            from tests.fakes import HashEmbedder

            fake = HashEmbedder()
            app.dependency_overrides[get_embedder_factory] = lambda: (lambda: fake)
        client = TestClient(app)

        up = client.post("/documents/upload", files={"file": ("verify.pdf", build_pdf(PAGES), "application/pdf")})
        check("upload 8-page PDF", up.status_code == 201 and up.json()["metadata"]["page_count"] == 8)
        doc = up.json()["document_id"]

        dry = client.post(f"/documents/{doc}/index?dry_run=true").json()
        check("dry run: 8 chunks, nothing spent", dry["chunk_count"] == 8 and dry["dry_run"],
              f"{dry['tokens']} tokens ~ ${dry['cost_usd']}")

        real = client.post(f"/documents/{doc}/index")
        check("index (embeds every chunk)", real.status_code == 200, f"HTTP {real.status_code}: {real.text[:200]}" if real.status_code != 200 else "")
        if real.status_code != 200:
            return
        rj = real.json()
        chunks = client.get(f"/documents/{doc}/chunks?limit=50").json()["chunks"]
        dims = {c["embedding_dim"] for c in chunks}
        check("all chunks have embeddings", all(c["has_embedding"] for c in chunks), f"dim={dims}")
        again = client.post(f"/documents/{doc}/index").json()
        check("re-index is free (cache)", again["tokens"] == 0 and again["chunks_from_cache"] == 8)
        total_tokens = rj["tokens"]

        modes = ["keyword"] if offline else ["keyword", "vector", "hybrid"]
        ranks: dict[str, list[int | None]] = {m: [] for m in modes}
        for question, page, _ in QUESTIONS:
            for mode in modes:
                r = client.post("/search", json={"query": question, "mode": mode, "top_k": 8, "document_id": doc})
                if r.status_code != 200:
                    check(f"search {mode}: '{question[:30]}'", False, f"HTTP {r.status_code} {r.text[:150]}")
                    ranks[mode].append(None)
                    continue
                body = r.json()
                total_tokens += body["query_tokens"]
                ranks[mode].append(rank_of(body["hits"], page))

        print("\n  Rank of the correct page (1 = best, - = not retrieved)")
        header = f"  {'question':<52} {'page':>4} " + " ".join(f"{m:>8}" for m in modes)
        print(header)
        print("  " + "-" * (len(header) - 2))
        for i, (question, page, note) in enumerate(QUESTIONS):
            cells = " ".join(f"{(ranks[m][i] or '-')!s:>8}" for m in modes)
            print(f"  {question[:50]:<52} {page:>4} {cells}")

        def hit_at(mode: str, k: int) -> int:
            return sum(1 for r in ranks[mode] if r is not None and r <= k)

        print()
        for m in modes:
            print(f"  {m:<8} hit@1 = {hit_at(m, 1)}/8   hit@3 = {hit_at(m, 3)}/8")
        print()

        check("keyword finds exact identifier 'ERR-4471' at rank 1", ranks["keyword"][0] == 1)
        if not offline:
            check("vector hit@3 >= 7/8", hit_at("vector", 3) >= 7, f"{hit_at('vector', 3)}/8")
            check("hybrid hit@3 == 8/8", hit_at("hybrid", 3) == 8, f"{hit_at('hybrid', 3)}/8")
            check("hybrid hit@1 >= 5/8", hit_at("hybrid", 1) >= 5, f"{hit_at('hybrid', 1)}/8")
            check("hybrid >= keyword-only (hit@3)", hit_at("hybrid", 3) >= hit_at("keyword", 3))
            q = QUESTIONS[1][0]
            cached = client.post("/search", json={"query": q, "mode": "vector", "document_id": doc}).json()
            check("repeated query is served from embedding cache", cached["query_tokens"] == 0)
            cost = total_tokens / 1_000_000 * settings.embedding_price_per_million_tokens
            print(f"\n  Total OpenAI usage this run: {total_tokens} tokens ~ ${cost:.6f}")
        app.dependency_overrides.clear()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="no OpenAI calls (keyword search only)")
    ap.add_argument("--skip-pytest", action="store_true")
    args = ap.parse_args()

    has_key = step_environment(args.offline)
    if not args.skip_pytest:
        step_pytest()
    if args.offline or has_key:
        step_end_to_end(offline=args.offline)
    else:
        print("\n[3/3] Skipped: no OPENAI_API_KEY (use --offline for a free keyword-only run)")

    failed = [r for r in RESULTS if not r[0]]
    print("\n" + "=" * 60)
    if failed:
        print(f"{len(failed)} CHECK(S) FAILED:")
        for _, name, detail in failed:
            print(f"  - {name} {detail}")
        return 1
    print(f"ALL {len(RESULTS)} CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
