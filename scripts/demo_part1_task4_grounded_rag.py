import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: F401,E402
from rag.retrieval import answer_or_fallback, calibrate_threshold  # noqa: E402
from scripts._demo_utils import Logger  # noqa: E402

IN_SCOPE_DEMO_QUERIES = [
    "What documents do I need for KYC verification?",
    "How is my EMI calculated?",
    "What fees apply to my credit card?",
    "Can I close my savings account anytime?",
    "Is there a penalty for prepaying my home loan early?",
]
OUT_OF_SCOPE_DEMO_QUERY = "What is the weather like in Paris today?"


def main():
    log = Logger()
    log("Part 1, Task 4: Empirical threshold calibration + grounded generation")

    calibration = calibrate_threshold()
    log("Calibration measurements (top-1 similarity):")
    for q, s in calibration["in_scope"]:
        log(f"  in-scope   {s:.4f}  {q}")
    for q, s in calibration["out_of_scope"]:
        log(f"  out-of-scope {s:.4f}  {q}")
    log(f"Lowest in-scope similarity: {calibration['min_in_scope']:.4f}")
    log(f"Highest out-of-scope similarity: {calibration['max_out_of_scope']:.4f}")
    log(f"Recommended threshold (midpoint): {calibration['recommended_threshold']:.4f}")
    log(f"config.SIMILARITY_THRESHOLD currently set to: {config.SIMILARITY_THRESHOLD}")
    log(
        "NOTE: if the recommended threshold above differs from "
        "config.SIMILARITY_THRESHOLD, update config.py's SIMILARITY_THRESHOLD "
        "to match and re-run this script, then cite the final numbers in README.md."
    )

    log("")
    log(f"Demonstrating on {len(IN_SCOPE_DEMO_QUERIES)} in-scope queries + 1 out-of-scope query:")
    for q in IN_SCOPE_DEMO_QUERIES:
        answer, is_fallback, retrieved = answer_or_fallback(q, "kb_sentence")
        log(f"Q: {q}")
        log(f"  top1_similarity={retrieved[0][1]:.4f} is_fallback={is_fallback}")
        log(f"  A: {answer}")

    answer, is_fallback, retrieved = answer_or_fallback(OUT_OF_SCOPE_DEMO_QUERY, "kb_sentence")
    top1 = retrieved[0][1] if retrieved else 0.0
    log(f"Q (out-of-scope): {OUT_OF_SCOPE_DEMO_QUERY}")
    log(f"  top1_similarity={top1:.4f} is_fallback={is_fallback}")
    log(f"  A: {answer}")
    assert is_fallback, "Out-of-scope query should have triggered the fallback"
    log("PASS: out-of-scope query correctly triggered the fallback.")

    log.save("part1_task4.txt")


if __name__ == "__main__":
    main()
