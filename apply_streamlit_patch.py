"""
apply_streamlit_patch.py -- adds a second frontend: a Streamlit app
(frontend/streamlit_app.py) that talks to your SAME FastAPI backend over
plain HTTP (via the `requests` library), with an Upload & Index tab, Ask
(with the same per-stage pipeline trace as the browser console), Search,
Chunks, and a live Monitor tab.

This is a pure addition -- it doesn't touch any existing file, and you
don't need the CORS change from before: CORS is a BROWSER rule, and this
is a Python process calling your API directly, same as curl.

Run this from inside your production-rag project folder:

    python apply_streamlit_patch.py

Then:
    pip install -r requirements-streamlit.txt
    streamlit run frontend/streamlit_app.py
"""
import argparse
import base64
import filecmp
import io
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

PAYLOAD = """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"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-install", action="store_true", help="skip pip install -r requirements-streamlit.txt")
    args = ap.parse_args()

    dest = Path.cwd()
    backup_dir = dest / "_backup_before_streamlit"

    raw = base64.b64decode(PAYLOAD)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            zf.extractall(tmp_path)

        new_count = replaced_count = unchanged_count = 0
        for src_file in sorted(tmp_path.rglob("*")):
            if src_file.is_dir():
                continue
            rel = src_file.relative_to(tmp_path)
            dst_file = dest / rel

            if not dst_file.exists():
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dst_file)
                new_count += 1
                continue

            if filecmp.cmp(src_file, dst_file, shallow=False):
                unchanged_count += 1
                continue

            backup_path = backup_dir / rel
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dst_file, backup_path)
            shutil.copy2(src_file, dst_file)
            replaced_count += 1

    print(f"Streamlit frontend added: {new_count} new, {replaced_count} replaced (backed up), "
          f"{unchanged_count} unchanged.")
    if replaced_count:
        print(f"Old versions saved in: {backup_dir}")

    if not args.no_install:
        print("\nInstalling requirements-streamlit.txt...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements-streamlit.txt"], check=False)
    else:
        print("\nSkipped install (--no-install). Run manually with: pip install -r requirements-streamlit.txt")

    print("\nStart it with:")
    print("  streamlit run frontend/streamlit_app.py")
    print("Your FastAPI server needs to be running separately (uvicorn app.main:app --reload --port 8000).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
