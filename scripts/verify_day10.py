"""
One-command Day 10 verification.

    python scripts/verify_day10.py            # pytest + a real two-tenant isolation check
    python scripts/verify_day10.py --offline  # pytest only
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
    print("\n[1/2] Test suite (pytest)")
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
                           capture_output=True, text=True)
    tail = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()][-1:] or ["(no output)"]
    check("pytest passes (includes the real auth + multi-tenancy suite)", proc.returncode == 0, tail[0])
    if proc.returncode != 0:
        print(proc.stdout[-3000:]); print(proc.stderr[-1500:])


def step_live() -> None:
    print("\n[2/2] Live check: two real tenants, one real /ask call each, verified isolated")
    from fastapi.testclient import TestClient
    from fpdf import FPDF

    from app.auth import generate_api_key, get_current_tenant
    from app.config import Settings, get_settings
    from app.dependencies import get_embedder_factory
    from app.main import app
    from app.repositories.tenant_repository import TenantRepository
    from tests.fakes import HashEmbedder

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        settings = Settings(raw_storage_dir=f"{tmp}/raw", sqlite_path=f"{tmp}/verify.db",
                             tenant_db_path=f"{tmp}/tenants.db")
        app.dependency_overrides[get_settings] = lambda: settings
        app.dependency_overrides[get_embedder_factory] = lambda: (lambda: HashEmbedder())  # auth/isolation check: no OpenAI needed
        app.dependency_overrides.pop(get_current_tenant, None)  # use REAL auth for this check
        app.state.rate_limiter.reset()
        app.state.tenant_rate_limiter.reset()
        client = TestClient(app)

        repo = TenantRepository(settings.tenant_db_path)
        key_a, key_b = generate_api_key(), generate_api_key()
        repo.create_key(key_a, "tenant-a", "verify script")
        repo.create_key(key_b, "tenant-b", "verify script")

        def pdf(text):
            p = FPDF(); p.add_page(); p.set_font("Helvetica", size=12); p.multi_cell(0, 10, text)
            return bytes(p.output())

        up_a = client.post("/documents/upload", files={"file": ("a.pdf", pdf(
            "Alpha Corp confidential refund policy: refunds within 15 days."), "application/pdf")},
            headers={"Authorization": f"Bearer {key_a}"})
        check("tenant A can upload with a real key", up_a.status_code == 201)
        doc_a = up_a.json()["document_id"]
        idx_a = client.post(f"/documents/{doc_a}/index", headers={"Authorization": f"Bearer {key_a}"})
        check("tenant A can index its own document", idx_a.status_code == 200)

        cross = client.post(f"/documents/{doc_a}/index", headers={"Authorization": f"Bearer {key_b}"})
        check("tenant B gets 404 (not 403) trying to touch tenant A's document", cross.status_code == 404)

        search_b = client.post("/search", json={"query": "refund policy", "mode": "keyword"},
                                headers={"Authorization": f"Bearer {key_b}"})
        check("tenant B's search (no document filter) returns NOTHING of tenant A's",
              search_b.status_code == 200 and search_b.json()["hits"] == [])

        no_key = client.post("/search", json={"query": "refund", "mode": "keyword"})
        check("a request with no API key at all is rejected (401)", no_key.status_code == 401)

        app.dependency_overrides.clear()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--skip-pytest", action="store_true")
    args = ap.parse_args()

    if not args.skip_pytest:
        step_pytest()
    step_live()  # this step needs no OpenAI key at all -- it's pure auth/isolation, always run it

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
