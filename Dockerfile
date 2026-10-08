# Day 18: containerizes the app exactly as it already runs locally --
# no code changes were needed to get here, because every storage path
# (app/storage/*) was already relative + self-creating (each repository's
# __init__ calls Path(path).parent.mkdir(parents=True, exist_ok=True)) and
# /health already has zero dependencies (see app/main.py). Containerizing
# an app that wasn't designed for it usually means discovering hardcoded
# absolute paths or hidden host dependencies; this one had none to find.
#
# Two stages: "builder" compiles hnswlib's C++ extension (needs gcc/g++,
# which the final image does not) into a user site-packages dir; "runtime"
# copies just that dir + the app code into a clean, smaller image with no
# compiler toolchain at all.
#
# Day 20: requirements.txt is now production-only (pytest/httpx/fpdf2 moved
# to requirements-dev.txt) -- the default build below installs ONLY what
# the running app needs. To build a variant that can also run the test
# suite INSIDE the container (what `docker compose exec api pytest` used
# to rely on through Day 19):
#   docker build --build-arg DEPS_FILE=requirements-dev.txt -t production-rag:dev .
#   docker run --rm production-rag:dev pytest

FROM python:3.13-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

ARG DEPS_FILE=requirements.txt

WORKDIR /app
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir --user -r ${DEPS_FILE}


FROM python:3.13-slim AS runtime

# Non-root by default -- the app never needs root (it only reads/writes
# its own app/storage/ tree and binds a port >1024), so there's no reason
# to run as one.
RUN groupadd --gid 1000 appuser && useradd --uid 1000 --gid appuser --create-home appuser

WORKDIR /app
COPY --from=builder /root/.local /home/appuser/.local
COPY . .

RUN chown -R appuser:appuser /app
USER appuser

ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

# No external tool (curl/wget) needed -- Python's own stdlib can hit
# /health, which app/main.py keeps deliberately dependency-free so this
# check answers instantly and never false-negatives on a slow downstream.
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=2)" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
