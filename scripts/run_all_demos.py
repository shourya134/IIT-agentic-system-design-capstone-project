"""Runs all 16 task demo scripts in sequence, regenerating every transcript.

Usage: python scripts/run_all_demos.py   (run from the repo root, inside the
activated venv, after `pip install -r requirements.txt` and after the
SentenceTransformers model has been downloaded once).
"""
import importlib
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: F401,E402

DEMO_MODULES = [
    "scripts.demo_part1_task1_dataset",
    "scripts.demo_part1_task2_kb",
    "scripts.demo_part1_task3_chunking",
    "scripts.demo_part1_task4_grounded_rag",
    "scripts.demo_part1_task5_eval_chunking",
    "scripts.demo_part2_task6_lookup",
    "scripts.demo_part2_task7_crew",
    "scripts.demo_part2_task8_memory",
    "scripts.demo_part2_task9_structured_output",
    "scripts.demo_part2_task10_guardrails",
    "scripts.demo_part3_task11_api",
    "scripts.demo_part3_task12_logging",
    "scripts.demo_part3_task13_judge",
    "scripts.demo_part4_task14_autogen_review",
    "scripts.demo_part4_task15_governance",
    "scripts.demo_part4_task16_caching",
]


def main():
    failures = []
    for module_name in DEMO_MODULES:
        print(f"\n{'=' * 80}\nRunning {module_name}\n{'=' * 80}")
        try:
            module = importlib.import_module(module_name)
            module.main()
        except Exception:
            print(f"FAILED: {module_name}")
            traceback.print_exc()
            failures.append(module_name)

    print(f"\n{'=' * 80}\nDone. {len(DEMO_MODULES) - len(failures)}/{len(DEMO_MODULES)} demos succeeded.")
    if failures:
        print("Failed demos:")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
