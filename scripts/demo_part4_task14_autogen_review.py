import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: F401,E402
from review.autogen_review import demo_approved_case, demo_revision_case  # noqa: E402
from scripts._demo_utils import Logger  # noqa: E402


def main():
    log = Logger()
    log("Part 4, Task 14: Autogen 2-agent RoundRobinGroupChat review stage")

    log("--- Case 1: draft has no unsupported claims -> approved unchanged ---")
    verdict1 = demo_approved_case()
    log(f"Verdict: {verdict1.model_dump_json()}")
    assert verdict1.approved is True
    log("PASS: draft approved unchanged.")

    log("--- Case 2: draft has a deliberately injected ungrounded claim -> revised ---")
    verdict2 = demo_revision_case()
    log(f"Verdict: {verdict2.model_dump_json()}")
    assert verdict2.approved is False
    log("PASS: draft caught and revised.")

    log.save("part4_task14.txt")


if __name__ == "__main__":
    main()
