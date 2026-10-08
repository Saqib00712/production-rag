"""
One-command Day 12 verification.

    python scripts/verify_day12.py            # pytest + a real /ask call on the hnsw backend + a small benchmark
    python scripts/verify_day12.py --offline  # pytest + benchmark only (no OpenAI call)

Note on what's asserted vs. just printed: wall-clock timing numbers are
machine-dependent and will vary run to run (CI runners especially) -- this
script asserts RECALL (a property of the algorithm/data, not the machine)
but only PRINTS latency/speedup for human inspection. Gating CI on an
absolute millisecond number is a recipe for flaky tests; gating on recall
is not.
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ.setdefault("LOG_LEVEL", "WARNING")

RESULTS: list[tuple[bool, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((ok, name, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return ok


def step_pytest() -> None:
    print("\n[1/3] Test suite (pytest)")
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
                           capture_output=True, text=True)
    tail = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()][-1:] or ["(no output)"]
    check("pytest passes (includes both vector-index backends)", proc.returncode == 0, tail[0])
    if proc.returncode != 0:
        print(proc.stdout[-3000:]); print(proc.stderr[-1500:])


def step_benchmark() -> None:
    print("\n[2/3] Small benchmark run: brute-force vs HNSW on clustered synthetic data")
    proc = subprocess.run(
        [sys.executable, "scripts/bench_vector_index.py", "--n-chunks", "3000", "--n-queries", "20",
         "--dim", "64", "--ef-search", "50"],
        capture_output=True, text=True,
    )
    print(proc.stdout)
    check("bench_vector_index.py runs without error", proc.returncode == 0)
    import re

    m = re.search(r"recall@10 vs\. brute-force ground truth: ([\d.]+)%", proc.stdout)
    if m:
        recall = float(m.group(1)) / 100
        check("HNSW recall@10 is high on clustered (realistic) data", recall >= 0.85, f"{recall:.2%}")
    else:
        check("could parse recall from benchmark output", False)


def step_live() -> None:
    print("\n[3/3] Live check: a real /ask call using the hnsw backend end-to-end")
    from fastapi.testclient import TestClient
    from fpdf import FPDF

    from app.auth import generate_api_key, get_current_tenant
    from app.config import Settings, get_settings
    from app.main import app
    from app.repositories.tenant_repository import TenantRepository

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        settings = Settings(raw_storage_dir=f"{tmp}/raw", sqlite_path=f"{tmp}/verify.db",
                             tenant_db_path=f"{tmp}/tenants.db", vector_index_backend="hnsw")
        app.dependency_overrides[get_settings] = lambda: settings
        app.dependency_overrides.pop(get_current_tenant, None)
        app.state.rate_limiter.reset(); app.state.tenant_rate_limiter.reset()
        client = TestClient(app)

        repo = TenantRepository(settings.tenant_db_path)
        key = generate_api_key()
        repo.create_key(key, "verify-day12", "verify script")
        headers = {"Authorization": f"Bearer {key}"}

        pdf = FPDF(); pdf.add_page(); pdf.set_font("Helvetica", size=12)
        pdf.multi_cell(0, 10, "Refund policy. Customers may request a refund within 30 days of purchase.")
        up = client.post("/documents/upload", files={"file": ("t.pdf", bytes(pdf.output()), "application/pdf")}, headers=headers)
        doc = up.json()["document_id"]
        idx = client.post(f"/documents/{doc}/index", headers=headers)
        check("indexing succeeds with vector_index_backend=hnsw configured", idx.status_code == 200)

        r = client.post("/ask", json={"question": "How many days for a refund?"}, headers=headers)
        body = r.json()
        print(f"  answer: {body.get('answer', '')[:150]!r}")
        check("real /ask call (whole-tenant search, hnsw backend) returns a grounded answer",
              r.status_code == 200 and len(body.get("citations", [])) > 0)
        app.dependency_overrides.clear()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--skip-pytest", action="store_true")
    args = ap.parse_args()

    if not args.skip_pytest:
        step_pytest()
    step_benchmark()

    if args.offline:
        print("\n[3/3] Skipped (--offline): live check needs a real OPENAI_API_KEY.")
    else:
        from app.config import Settings

        key = Settings().openai_api_key
        if not key or key.startswith("sk-your"):
            print("\n[3/3] Skipped: no OPENAI_API_KEY (use --offline to skip this deliberately)")
        else:
            step_live()

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
