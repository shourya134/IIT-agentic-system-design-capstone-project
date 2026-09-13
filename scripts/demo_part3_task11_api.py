import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: F401,E402
from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402
from scripts._demo_utils import Logger  # noqa: E402


def main():
    log = Logger()
    log("Part 3, Task 11: FastAPI endpoints (2 HTTP + 1 WebSocket) via TestClient")

    client = TestClient(app)

    log("--- POST /ask ---")
    resp = client.post("/ask", json={"query": "What documents do I need for KYC verification?"})
    log(f"status_code={resp.status_code}")
    log(f"body={resp.json()}")
    assert resp.status_code == 200

    log("--- POST /add-document ---")
    resp = client.post(
        "/add-document",
        json={
            "doc_id": "demo_extra_topic",
            "text": "Demo-only addendum. This is a test document added at runtime. It covers a hypothetical new policy topic.",
            "strategy": "both",
        },
    )
    log(f"status_code={resp.status_code}")
    log(f"body={resp.json()}")
    assert resp.status_code == 200

    log("--- WS /ws/chat: normal multi-turn exchange, then a disconnect ---")
    with client.websocket_connect("/ws/chat") as ws:
        ws.send_text("My record id is LN0002.")
        reply1 = ws.receive_text()
        log(f"reply1={reply1}")
        ws.send_text("What is my record id?")
        reply2 = ws.receive_text()
        log(f"reply2={reply2}")
    log("First client disconnected cleanly (context manager closed the socket).")

    log("--- WS /ws/chat: a SECOND client connects after the first disconnected ---")
    with client.websocket_connect("/ws/chat") as ws2:
        ws2.send_text("Hello again")
        reply3 = ws2.receive_text()
        log(f"reply3={reply3}")
    log("PASS: server kept serving a new client after the previous one disconnected (WebSocketDisconnect handled gracefully).")

    log.save("part3_task11.txt")


if __name__ == "__main__":
    main()
