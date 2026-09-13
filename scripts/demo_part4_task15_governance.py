import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: F401,E402
from governance.budget import BudgetExceededError, BudgetGuard  # noqa: E402
from governance.governance import GOVERNANCE_LAYERS, assert_tool_ownership, classify_risk  # noqa: E402
from scripts._demo_utils import Logger  # noqa: E402


def main():
    log = Logger()
    log("Part 4, Task 15: 4-layer governance, least-autonomy enforcement, runtime budget cap")

    log("--- Application layer: least-autonomy enforcement ---")
    for line in assert_tool_ownership():
        log(line)

    log("--- Risk classification ---")
    risk = classify_risk()
    log(f"Level: {risk['level']}")
    log(f"Justification: {risk['justification']}")

    log("--- 4-layer governance model ---")
    for layer, desc in GOVERNANCE_LAYERS.items():
        log(f"  {layer}: {desc}")

    log("--- Runtime layer: per-request budget cap ---")
    guard = BudgetGuard()
    normal_query = "What is the status of application LN0001?"
    total = guard.check(normal_query)
    log(f"Normal request estimated total: {total} tokens -> accepted")

    oversized_query = "Please explain loan policy in extreme detail. " * 2000
    try:
        guard.check(oversized_query)
        log("ERROR: oversized request was NOT rejected")
    except BudgetExceededError as e:
        log(f"Oversized request correctly rejected: {e}")

    log.save("part4_task15.txt")


if __name__ == "__main__":
    main()
