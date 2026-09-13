"""Structured JSON-Lines request logging with trace IDs (Part 3, Task 12).

The same PII masking used by the input guardrail (Task 10) is applied to
every string field before it is written to disk, so a fixed-format PII field
(PAN/Aadhaar/bank account number) never reaches the log file unmasked, even
if it slipped through as part of a request or response payload.
"""
import json
import time
import uuid
from pathlib import Path

import config
from crew.guardrails import mask_pii


def new_trace_id() -> str:
    return str(uuid.uuid4())


def _mask_value(value):
    if isinstance(value, str):
        masked, _ = mask_pii(value)
        return masked
    if isinstance(value, dict):
        return {k: _mask_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_mask_value(v) for v in value]
    return value


def log_request(
    trace_id: str,
    endpoint: str,
    request_payload: dict,
    response_payload: dict,
    duration_ms: float,
    log_path: str = config.LOGS_PATH,
) -> dict:
    entry = {
        "trace_id": trace_id,
        "timestamp": time.time(),
        "endpoint": endpoint,
        "duration_ms": round(duration_ms, 2),
        "request": _mask_value(request_payload),
        "response": _mask_value(response_payload),
    }
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


class Timer:
    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.duration_ms = (time.perf_counter() - self._start) * 1000
