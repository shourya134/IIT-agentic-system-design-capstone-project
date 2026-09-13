import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: F401,E402
from data.dataset import LOAN_APPLICATIONS, print_summary, save_dataset  # noqa: E402
from scripts._demo_utils import TRANSCRIPTS_DIR  # noqa: E402


def main():
    header = "Part 1, Task 1: Deterministic loan-application dataset"
    print(header)
    summary_lines = print_summary(LOAN_APPLICATIONS)
    save_dataset(LOAN_APPLICATIONS)
    footer = f"Saved {len(LOAN_APPLICATIONS)} records to {config.DATASET_PATH}"
    print(footer)

    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    (TRANSCRIPTS_DIR / "part1_task1.txt").write_text(
        "\n".join([header] + summary_lines + [footer]) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
