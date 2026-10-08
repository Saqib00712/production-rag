"""
One-command Day 6 verification.

    python scripts/verify_day6.py            # pytest + a live mini tuning run (a few cents)
    python scripts/verify_day6.py --offline  # pytest only (the tuning logic has 10 unit tests; the
                                              # live sweep needs a real index + real API calls, so
                                              # --offline skips step 2 entirely rather than faking it)
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
    check("pytest passes (includes the offline retrieval regression gate)", proc.returncode == 0, tail[0])
    if proc.returncode != 0:
        print(proc.stdout[-3000:]); print(proc.stderr[-1500:])


def step_live_tune() -> None:
    print("\n[2/2] Live mini tuning run (a small real sweep, a few cents at most)")
    import subprocess as sp

    from fpdf import FPDF

    PAGES = [
        "Refund policy. Customers may request a refund within 30 days of purchase, provided the item is unused.",
        "Warranty terms. The device carries a two year limited warranty covering manufacturing defects. Claims require serial number ERR-4471.",
        "Shipping information. Standard shipping takes five to seven business days within the country.",
        "Privacy notice. We never sell personal data. Users may request deletion of their account information at any time.",
    ]
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        pdf = FPDF()
        for t in PAGES:
            pdf.add_page(); pdf.set_font("Helvetica", size=12); pdf.multi_cell(0, 10, t)
        pdf_path = Path(tmp) / "verify.pdf"
        pdf_path.write_bytes(bytes(pdf.output()))

        golden_path = Path(tmp) / "golden.json"
        golden_path.write_text("""
        [
          {"id": "q1", "question": "How many days for a refund?", "expected_pages": [1]},
          {"id": "q2", "question": "What serial number is needed for warranty?", "expected_pages": [2]},
          {"id": "q3", "question": "How long does shipping take?", "expected_pages": [3]},
          {"id": "q4", "question": "Can personal data be deleted?", "expected_pages": [4]},
          {"id": "q5", "question": "What is the CEO's annual bonus?", "expected_pages": []}
        ]
        """, encoding="utf-8")

        proc = sp.run(
            [sys.executable, "scripts/tune.py", "--pdf", str(pdf_path), "--golden", str(golden_path),
             "--chunk-sizes", "150", "300", "--retrieve-ks", "5", "10", "--skip-threshold"],
            capture_output=True, text=True,
        )
        ok = proc.returncode == 0 and "recommended_chunk_size" not in proc.stderr
        check("tune.py runs end-to-end without error", proc.returncode == 0,
              proc.stderr[-500:] if proc.returncode != 0 else "")
        if proc.returncode == 0:
            print("\n".join("  " + ln for ln in proc.stdout.splitlines() if ln.strip()))
            check("chunk sweep produced a recommendation", "-> best: chunk_size=" in proc.stdout)
            check("retrieve_k sweep produced a recommendation", "-> smallest k reaching" in proc.stdout)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--skip-pytest", action="store_true")
    args = ap.parse_args()

    if not args.skip_pytest:
        step_pytest()

    if args.offline:
        print("\n[2/2] Skipped (--offline): the tuning sweep needs a real index and real API calls; "
              "its logic is covered by tests/test_tuning.py (10 tests) instead.")
    else:
        from app.config import Settings

        key = Settings().openai_api_key
        if not key or key.startswith("sk-your"):
            print("\n[2/2] Skipped: no OPENAI_API_KEY (use --offline to skip this step deliberately)")
        else:
            step_live_tune()

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
