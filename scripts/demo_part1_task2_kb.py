import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: F401,E402
from rag.chunking import load_kb_docs  # noqa: E402
from scripts._demo_utils import Logger  # noqa: E402

REQUIRED_TOPICS = [
    "loan_eligibility", "emi_calculation", "credit_card_fees", "kyc_requirements",
    "fraud_dispute_resolution", "account_closure", "interest_rate_slabs",
    "prepayment_penalty", "minimum_balance", "credit_score_factors",
    "joint_account_rules", "nri_account_eligibility",
]


def main():
    log = Logger()
    log("Part 1, Task 2: Knowledge-base document coverage")
    docs = load_kb_docs()
    log(f"Loaded {len(docs)} documents: {sorted(docs.keys())}")
    missing = [t for t in REQUIRED_TOPICS if t not in docs]
    log(f"Required topics covered: {len(REQUIRED_TOPICS) - len(missing)}/{len(REQUIRED_TOPICS)}")
    if missing:
        log(f"MISSING TOPICS: {missing}")
    else:
        log("All 12 required topics are present.")
    for doc_id, text in docs.items():
        sentence_count = text.count(". ") + text.count("? ") + text.count("! ") + 1
        log(f"  {doc_id}: ~{sentence_count} sentences, {len(text)} chars")
    log.save("part1_task2.txt")


if __name__ == "__main__":
    main()
