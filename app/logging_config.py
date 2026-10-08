"""
Structured logging setup.

Design decision: every log line is a JSON object with consistent fields.
This is what lets you query logs later ("show me every ingestion error in
the last hour") instead of grepping free-text sentences. In production this
JSON stream is shipped to something like CloudWatch, Loki, or Datadog.

We deliberately do NOT use print() anywhere in this codebase from Day 1
onward -- print() has no level, no timestamp, no request context, and is
easy to forget to remove.
"""

import logging
import sys
import json
import time
from app.config import get_settings


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": time.strftime(
                "%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Extra structured fields attached via logger.info(..., extra={...})
        for key in ("request_id", "document_id", "duration_ms", "error_type",
                    "chunks", "tokens", "cost_usd", "mode", "hits", "refusal_reason",
                    "method", "path", "status_code", "client_ip", "pii_types", "spans"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging() -> None:
    settings = get_settings()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(settings.log_level)

    # Quiet noisy third-party loggers so our structured signal isn't drowned
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
