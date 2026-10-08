"""
One-command Day 8 verification.

    python scripts/verify_day8.py            # pytest + a real streamed /ask call
    python scripts/verify_day8.py --offline  # pytest only
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
    check("pytest passes (includes the streaming pipeline + SSE endpoint suites)", proc.returncode == 0, tail[0])
    if proc.returncode != 0:
        print(proc.stdout[-3000:]); print(proc.stderr[-1500:])


def step_live() -> None:
    print("\n[2/2] Live streamed /ask call against your real API key")
    import json
    import time

    from fastapi.testclient import TestClient
    from fpdf import FPDF

    from app.config import Settings, get_settings
    from app.main import app

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        settings = Settings(raw_storage_dir=f"{tmp}/raw", sqlite_path=f"{tmp}/verify.db")
        app.dependency_overrides[get_settings] = lambda: settings
        app.state.rate_limiter.reset()
        client = TestClient(app)

        pdf = FPDF()
        pdf.add_page(); pdf.set_font("Helvetica", size=12)
        pdf.multi_cell(0, 10, "Refund policy. Customers may request a refund within 30 days of purchase.")
        up = client.post("/documents/upload", files={"file": ("t.pdf", bytes(pdf.output()), "application/pdf")})
        doc = up.json()["document_id"]
        client.post(f"/documents/{doc}/index")

        deltas = []
        events_seen = []
        start = time.perf_counter()
        first_delta_at = None
        with client.stream("POST", "/ask/stream",
                            json={"question": "How many days for a refund?", "document_id": doc}) as r:
            check("HTTP 200 with text/event-stream content-type", r.status_code == 200
                  and r.headers.get("content-type", "").startswith("text/event-stream"))
            buffer = ""
            for chunk in r.iter_text():
                buffer += chunk
                while "\n\n" in buffer:
                    block, buffer = buffer.split("\n\n", 1)
                    lines = block.strip().split("\n")
                    if not lines or not lines[0]:
                        continue
                    event_type = lines[0].split(": ", 1)[1]
                    data = json.loads(lines[1].split(": ", 1)[1])
                    events_seen.append(event_type)
                    if event_type == "delta":
                        if first_delta_at is None:
                            first_delta_at = time.perf_counter() - start
                        deltas.append(data["text"])
                    elif event_type == "done":
                        done_data = data

        full_text = "".join(deltas)
        print(f"  received {len(deltas)} delta chunk(s), first arrived after {first_delta_at:.2f}s" if first_delta_at else "  no deltas received")
        print(f"  streamed text: {full_text[:150]!r}")
        check("received at least one delta chunk", len(deltas) > 0)
        check("events end with a 'done' event", events_seen and events_seen[-1] == "done")
        check("final answer is grounded with a citation, or was correctly flagged for correction",
              len(done_data.get("citations", [])) > 0 or "correction" in events_seen)
        if first_delta_at is not None:
            # No fixed pass/fail threshold here: reranking is a full, real LLM
            # call that MUST complete before generation can start (the system
            # doesn't know which sources to hand the generator until reranking
            # has ranked them) -- so "time to first token" includes retrieval
            # + a full rerank round-trip, not just generation. That can
            # legitimately take several seconds depending on network and
            # provider load. This is printed for visibility, not gated.
            print(f"  time to first token: {first_delta_at:.2f}s (includes retrieval + a full rerank call, "
                  f"which must finish before streaming generation can begin)")
        app.dependency_overrides.clear()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--skip-pytest", action="store_true")
    args = ap.parse_args()

    if not args.skip_pytest:
        step_pytest()

    if args.offline:
        print("\n[2/2] Skipped (--offline): live streaming needs a real OPENAI_API_KEY.")
    else:
        from app.config import Settings

        key = Settings().openai_api_key
        if not key or key.startswith("sk-your"):
            print("\n[2/2] Skipped: no OPENAI_API_KEY (use --offline to skip this deliberately)")
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
