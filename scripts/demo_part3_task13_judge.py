import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: F401,E402
from eval.judge import print_report, run_eval  # noqa: E402
from scripts._demo_utils import TRANSCRIPTS_DIR  # noqa: E402


def main():
    header = "Part 3, Task 13: LLM-as-judge evaluation across 15 queries (Accuracy/Grounding/Completeness/Safety)"
    print(header)
    rows = run_eval()
    body_lines = print_report(rows)

    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    (TRANSCRIPTS_DIR / "part3_task13.txt").write_text(
        "\n".join([header, ""] + body_lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
