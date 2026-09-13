"""Deterministic, seeded loan-application dataset generator (Part 1, Task 1).

Design notes (see README.md for the full write-up):

- Category and status coverage is GUARANTEED BY CONSTRUCTION, not by
  probabilistic luck: we build a round-robin sequence of each field (so every
  value appears roughly n/5 times), then shuffle each sequence independently
  with its own seeded RNG. This means the ">=3 per category" / ">=1 per
  status" constraints hold for any n >= 25 regardless of seed, so we never
  have to "check and hope" or hand-edit individual records.
- `loan_amount_inr` range is chosen per category to be realistic for the
  Indian lending market (see AMOUNT_RANGES_INR below for the one-sentence
  justification per category, as required by the spec).
- `flagged_for_fraud_review` uses a single fixed probability (FRAUD_FLAG_PROB)
  applied independently per record. This constant was chosen, then the
  resulting percentage was measured by running this script -- it was NOT
  reverse-engineered by hand-editing individual flags after the fact. If you
  change the seed or n, re-run `python data/dataset.py` and confirm the
  printed percentage still falls in [10%, 30%]; if it doesn't, adjust
  FRAUD_FLAG_PROB (not individual records) and re-run.
"""
import json
import random
from collections import Counter
from pathlib import Path

import config

CATEGORIES = ["Personal Loan", "Home Loan", "Auto Loan", "Education Loan", "Business Loan"]
STATUSES = ["Submitted", "Under Review", "Approved", "Rejected", "Disbursed"]

# One-sentence justification per category, cited verbatim in README.md:
# Personal Loan: unsecured, small-ticket, income-driven -> kept low relative to secured loans.
# Home Loan: secured against property, India's largest ticket-size loan category by far.
# Auto Loan: secured against the vehicle, mid-range ticket size tied to vehicle price bands.
# Education Loan: covers tuition + living costs for domestic/overseas study, wide range.
# Business Loan: working-capital/expansion financing for SMEs, wide range by business size.
AMOUNT_RANGES_INR = {
    "Personal Loan": (50_000, 1_500_000),
    "Home Loan": (1_000_000, 20_000_000),
    "Auto Loan": (200_000, 2_500_000),
    "Education Loan": (100_000, 4_000_000),
    "Business Loan": (500_000, 10_000_000),
}

FRAUD_FLAG_PROB = 0.18
N_RECORDS = 48


def _round_robin_sequence(values: list[str], n: int, seed: int) -> list[str]:
    sequence = [values[i % len(values)] for i in range(n)]
    random.Random(seed).shuffle(sequence)
    return sequence


def generate_dataset(seed: int = config.RNG_SEED, n: int = N_RECORDS) -> list[dict]:
    category_sequence = _round_robin_sequence(CATEGORIES, n, seed)
    status_sequence = _round_robin_sequence(STATUSES, n, seed + 1)

    amount_rng = random.Random(seed + 2)
    days_rng = random.Random(seed + 3)
    fraud_rng = random.Random(seed + 4)

    records = []
    for i in range(n):
        category = category_sequence[i]
        status = status_sequence[i]
        lo, hi = AMOUNT_RANGES_INR[category]
        record = {
            "record_id": f"LN{i + 1:04d}",
            "category": category,
            "status": status,
            "loan_amount_inr": amount_rng.randint(lo, hi),
            "days_since_created": days_rng.randint(0, 30),
            "flagged_for_fraud_review": fraud_rng.random() < FRAUD_FLAG_PROB,
        }
        records.append(record)
    return records


def print_summary(records: list[dict]) -> list[str]:
    lines = []
    category_counts = Counter(r["category"] for r in records)
    status_counts = Counter(r["status"] for r in records)
    fraud_count = sum(1 for r in records if r["flagged_for_fraud_review"])
    fraud_pct = 100.0 * fraud_count / len(records)

    lines.append(f"Total records: {len(records)}")
    lines.append("Category counts:")
    for cat in CATEGORIES:
        lines.append(f"  {cat}: {category_counts.get(cat, 0)}")
    lines.append("Status counts:")
    for st in STATUSES:
        lines.append(f"  {st}: {status_counts.get(st, 0)}")
    lines.append(
        f"Fraud-flagged: {fraud_count}/{len(records)} = {fraud_pct:.1f}% "
        f"(must be in [10%, 30%])"
    )
    for line in lines:
        print(line)
    return lines


def save_dataset(records: list[dict], path: str = config.DATASET_PATH) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(records, indent=2), encoding="utf-8")


def get_record(record_id: str, records: list[dict] | None = None) -> dict | None:
    records = records if records is not None else LOAN_APPLICATIONS
    for r in records:
        if r["record_id"] == record_id:
            return r
    return None


LOAN_APPLICATIONS = generate_dataset()


if __name__ == "__main__":
    print_summary(LOAN_APPLICATIONS)
    save_dataset(LOAN_APPLICATIONS)
