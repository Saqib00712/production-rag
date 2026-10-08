"""
One-command Day 11 verification.

    python scripts/verify_day11.py            # pytest + validate CI config + a real cost-bounded eval
    python scripts/verify_day11.py --offline  # pytest + validate CI config only
"""

import argparse
import os
import subprocess
import sys
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
    check("pytest passes (includes the cost-gate suite)", proc.returncode == 0, tail[0])
    if proc.returncode != 0:
        print(proc.stdout[-3000:]); print(proc.stderr[-1500:])


def step_validate_ci_config() -> None:
    print("\n[2/3] CI configuration sanity checks")
    workflow_path = Path(".github/workflows/ci.yml")
    check(".github/workflows/ci.yml exists", workflow_path.exists())
    if not workflow_path.exists():
        return
    try:
        import yaml

        doc = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
        check("ci.yml is valid YAML", True)
    except Exception as exc:
        check("ci.yml is valid YAML", False, str(exc))
        return
    jobs = doc.get("jobs", {})
    check("workflow defines a 'test' job", "test" in jobs)
    check("workflow defines a 'live-eval' job", "live-eval" in jobs)
    if "live-eval" in jobs:
        condition = jobs["live-eval"].get("if", "")
        check("'live-eval' is gated to push-to-main (not every PR)", "refs/heads/main" in condition, condition)
    check("scripts/ci_eval.py exists (what the live-eval job runs)", Path("scripts/ci_eval.py").exists())
    check("scripts/install_git_hooks.py exists (optional local pre-push gate)",
          Path("scripts/install_git_hooks.py").exists())


def step_live() -> None:
    print("\n[3/3] Live check: the exact command CI runs, with a tight cost ceiling")
    proc = subprocess.run([sys.executable, "scripts/ci_eval.py", "--max-cost-usd", "0.05"],
                           capture_output=True, text=True)
    print(proc.stdout)
    if proc.stderr.strip():
        print(proc.stderr[-1000:])
    check("scripts/ci_eval.py exits 0 (within cost ceiling and quality thresholds, or skipped cleanly)",
          proc.returncode == 0, f"exit code {proc.returncode}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--skip-pytest", action="store_true")
    args = ap.parse_args()

    if not args.skip_pytest:
        step_pytest()
    step_validate_ci_config()

    if args.offline:
        print("\n[3/3] Skipped (--offline): live cost-bounded eval needs a real OPENAI_API_KEY.")
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
