import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: F401,E402
from memory.memory_chain import demo_fresh_session, demo_multiturn  # noqa: E402
from scripts._demo_utils import TRANSCRIPTS_DIR  # noqa: E402


def main():
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    multiturn_lines = [
        "Part 2, Task 8: multi-turn session memory (state CARRIED across turns)",
        "",
    ] + demo_multiturn()
    for line in multiturn_lines:
        print(line)
    (TRANSCRIPTS_DIR / "part2_task8_multiturn.txt").write_text(
        "\n".join(multiturn_lines) + "\n", encoding="utf-8"
    )

    fresh_lines = [
        "Part 2, Task 8: SEPARATE fresh-conversation transcript (state correctly ABSENT)",
        "",
    ] + demo_fresh_session()
    for line in fresh_lines:
        print(line)
    (TRANSCRIPTS_DIR / "part2_task8_fresh.txt").write_text(
        "\n".join(fresh_lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
