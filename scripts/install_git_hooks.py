"""
Installs a local pre-push git hook that runs the $0 test suite before
`git push` is allowed to proceed -- a personal, local-only safety net that
catches a broken commit BEFORE it reaches GitHub Actions, not a replacement
for CI (CI is still the authoritative gate; this just saves you the round
trip of pushing, waiting, and seeing a red X).

    python scripts/install_git_hooks.py           # install
    python scripts/install_git_hooks.py --remove  # uninstall

Works on Windows via Git Bash (which ships with Git for Windows and is what
actually executes git hooks, even on Windows) as well as macOS/Linux.
"""

import argparse
import os
import stat
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HOOK_PATH = ROOT / ".git" / "hooks" / "pre-push"

HOOK_SCRIPT = """#!/bin/sh
# Installed by scripts/install_git_hooks.py -- runs the test suite before
# allowing a push. Remove with: python scripts/install_git_hooks.py --remove
echo "Running test suite before push (pre-push hook)..."
pytest -q
status=$?
if [ $status -ne 0 ]; then
    echo ""
    echo "Tests failed -- push aborted. Fix the failure, or bypass with 'git push --no-verify'."
    exit 1
fi
exit 0
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--remove", action="store_true")
    args = ap.parse_args()

    if not (ROOT / ".git").is_dir():
        print("ERROR: no .git directory found here -- run this from inside your git repository, "
              "and make sure you've run 'git init' first.")
        return 1

    if args.remove:
        if HOOK_PATH.exists():
            HOOK_PATH.unlink()
            print(f"Removed {HOOK_PATH}")
        else:
            print("No pre-push hook was installed.")
        return 0

    HOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    HOOK_PATH.write_text(HOOK_SCRIPT, newline="\n", encoding="utf-8")
    if os.name != "nt":
        HOOK_PATH.chmod(HOOK_PATH.stat().st_mode | stat.S_IEXEC)
    print(f"Installed pre-push hook at {HOOK_PATH}")
    print("Every 'git push' will now run 'pytest -q' first. Bypass once with 'git push --no-verify'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
