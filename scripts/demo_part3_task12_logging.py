import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: F401,E402
from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402
from scripts._demo_utils import Logger  # noqa: E402

LOG_PATH = Path(config.LOGS_PATH)


def main():
    log = Logger()
    log("Part 3, Task 12: structured JSON-Lines logging with trace ID + timing, PII masked before write")

    # for clearing up any previous log entries from prior runs of this demo script
    if LOG_PATH.exists():
        LOG_PATH.unlink()

    #initialize the test client and send a request that contains a raw PAN number (which should be masked before logging)
    client = TestClient(app)
    pan_query = "My PAN is ABCDE1234F -- what documents do I need for KYC verification?"
    resp = client.post("/ask", json={"query": pan_query})
    log(f"Request sent (contains a raw PAN number): {pan_query!r}")
    log(f"Response status: {resp.status_code}")

    
    assert LOG_PATH.exists(), "Expected a log entry to have been written"
    entries = [json.loads(line) for line in LOG_PATH.read_text(encoding="utf-8").splitlines()]
    last_entry = entries[-1]
    log(f"Logged entry: {json.dumps(last_entry, indent=2)}")

    assert "trace_id" in last_entry and last_entry["trace_id"]
    assert "duration_ms" in last_entry
    logged_text = json.dumps(last_entry)
    assert "ABCDE1234F" not in logged_text, "Raw PAN number must never reach the log file"
    log("PASS: log entry has a trace_id and duration_ms, and the raw PAN number is not present anywhere in the logged JSON.")

    log.save("part3_task12.txt")


if __name__ == "__main__":
    main()
