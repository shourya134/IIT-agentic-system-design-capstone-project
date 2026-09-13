import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: F401,E402
from rag.evaluation import compare_strategies  # noqa: E402
from scripts._demo_utils import TRANSCRIPTS_DIR  # noqa: E402


def main():
    lines = ["Part 1, Task 5: Precision/recall comparison across both chunking strategies", ""]
    lines.extend(compare_strategies())
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    (TRANSCRIPTS_DIR / "part1_task5.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
