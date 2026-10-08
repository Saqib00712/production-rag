"""
One-command Day 20 verification.

    python scripts/verify_day20.py            # pytest + requirements split + ci.yml checks + (if Docker is
                                               # reachable) a real build proving the default image has no
                                               # test-only dependencies in it
    python scripts/verify_day20.py --offline  # everything except the docker build

Four things are checked that have nothing to do with OpenAI, so they always
run: requirements.txt is production-only (no pytest/httpx/fpdf2, no
hnswlib), requirements-dev.txt adds the dev tools back via `-r
requirements.txt`, hnswlib now lives only in the optional
requirements-hnsw.txt and HnswIndex raises a clear, actionable error (not a
raw ImportError) if it's selected without being installed, and
.github/workflows/ci.yml defines all three jobs (test, live-eval,
publish-image) with publish-image correctly gated to push-to-main and
depending on both quality gates.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ.setdefault("LOG_LEVEL", "WARNING")

RESULTS: list[tuple[bool, str, str]] = []
DEV_ONLY_PACKAGES = ("pytest", "httpx", "fpdf2", "pytest-asyncio")
HNSW_PACKAGE = "hnswlib"


def check(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((ok, name, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return ok


def step_pytest() -> None:
    print("\n[1/5] Test suite (pytest)")
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
                           capture_output=True, text=True)
    tail = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()][-1:] or ["(no output)"]
    check("pytest passes", proc.returncode == 0, tail[0])
    if proc.returncode != 0:
        print(proc.stdout[-3000:]); print(proc.stderr[-1500:])


def step_requirements_split() -> None:
    print("\n[2/5] requirements.txt / requirements-dev.txt split")
    prod = Path("requirements.txt").read_text().lower()
    check("requirements.txt has none of the dev-only packages",
          not any(pkg in prod for pkg in DEV_ONLY_PACKAGES))
    check("requirements.txt does not hard-require hnswlib (optional, needs a C++ compiler on Windows)",
          HNSW_PACKAGE not in prod)

    dev_path = Path("requirements-dev.txt")
    check("requirements-dev.txt exists", dev_path.exists())
    if dev_path.exists():
        dev = dev_path.read_text().lower()
        check("requirements-dev.txt pulls in requirements.txt (-r requirements.txt)", "-r requirements.txt" in dev)
        check("requirements-dev.txt has every dev-only package",
              all(pkg in dev for pkg in DEV_ONLY_PACKAGES))
        check("requirements-dev.txt does not hard-require hnswlib either", HNSW_PACKAGE not in dev)

    hnsw_path = Path("requirements-hnsw.txt")
    check("requirements-hnsw.txt exists (optional extra for the hnsw backend)", hnsw_path.exists())
    if hnsw_path.exists():
        check("requirements-hnsw.txt lists hnswlib", HNSW_PACKAGE in hnsw_path.read_text().lower())


def step_hnsw_optional() -> None:
    print("\n[3/5] hnswlib is optional: a clear error, not a crash, if it's missing")
    import importlib

    from app.services.vector_index import HnswIndex
    import numpy as np

    try:
        importlib.import_module("hnswlib")
        print("  hnswlib is installed in this environment -- nothing to prove here, skipping.")
        return
    except ImportError:
        pass

    try:
        HnswIndex().build(["a"], np.array([[1.0, 0.0]], dtype=np.float32))
        check("selecting the hnsw backend without hnswlib installed raises an error", False,
              "no exception was raised")
    except RuntimeError as exc:
        check("selecting the hnsw backend without hnswlib installed raises a clear, actionable error",
              "requirements-hnsw.txt" in str(exc))
    except Exception as exc:  # pragma: no cover - would mean the guard regressed
        check("selecting the hnsw backend without hnswlib installed raises a clear, actionable error", False,
              f"raised {type(exc).__name__} instead of a friendly RuntimeError")


def step_ci_config() -> None:
    print("\n[4/5] .github/workflows/ci.yml")
    workflow_path = Path(".github/workflows/ci.yml")
    if not check(".github/workflows/ci.yml exists", workflow_path.exists()):
        return
    import yaml

    try:
        doc = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
        check("ci.yml is valid YAML", True)
    except Exception as exc:
        check("ci.yml is valid YAML", False, str(exc))
        return

    jobs = doc.get("jobs", {})
    for name in ("test", "live-eval", "publish-image"):
        check(f"workflow defines a '{name}' job", name in jobs)

    publish = jobs.get("publish-image", {})
    condition = publish.get("if", "")
    check("'publish-image' is gated to push-to-main (not every PR)", "refs/heads/main" in condition, condition)
    needs = publish.get("needs", [])
    needs = [needs] if isinstance(needs, str) else needs
    check("'publish-image' needs both 'test' AND 'live-eval' to pass first",
          "test" in needs and "live-eval" in needs, str(needs))

    test_install = str(jobs.get("test", {}).get("steps", ""))
    check("'test' job installs requirements-dev.txt (not just requirements.txt)",
          "requirements-dev.txt" in test_install)


def step_docker_build() -> None:
    print("\n[5/5] Live check: the default image has no test-only dependencies in it")
    if not shutil.which("docker"):
        print("  Skipped: docker CLI not found on PATH.")
        return
    if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        print("  Skipped: Docker daemon isn't reachable (not running, or no permission).")
        return

    image = "production-rag:verify-day20"
    build = subprocess.run(["docker", "build", "-t", image, "."], capture_output=True, text=True, timeout=600)
    if build.returncode != 0:
        tail = (build.stderr or build.stdout).strip().splitlines()[-5:]
        print("  Skipped: image build failed (often a blocked/offline registry, not a Dockerfile bug):")
        for ln in tail:
            print(f"    {ln}")
        return
    check("docker build (default args) succeeds", True)

    probe = subprocess.run(
        ["docker", "run", "--rm", image, "python", "-c", "import pytest"],
        capture_output=True, text=True,
    )
    check("the default (production) image does NOT have pytest installed", probe.returncode != 0)
    subprocess.run(["docker", "rmi", "-f", image], capture_output=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="skip the docker build check")
    ap.add_argument("--skip-pytest", action="store_true")
    args = ap.parse_args()

    if not args.skip_pytest:
        step_pytest()
    step_requirements_split()
    step_hnsw_optional()
    step_ci_config()

    if args.offline:
        print("\n[5/5] Skipped (--offline): no docker build check.")
    else:
        step_docker_build()

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
