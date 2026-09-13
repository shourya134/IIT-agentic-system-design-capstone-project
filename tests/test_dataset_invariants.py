"""Verifies the Task 1 structural dataset guarantees hold for the actual
generated dataset (not just "should hold in theory")."""
from collections import Counter

from data.dataset import CATEGORIES, LOAN_APPLICATIONS, STATUSES


def test_at_least_40_records():
    assert len(LOAN_APPLICATIONS) >= 40


def test_every_category_has_at_least_3():
    counts = Counter(r["category"] for r in LOAN_APPLICATIONS)
    for category in CATEGORIES:
        assert counts.get(category, 0) >= 3, f"{category} has fewer than 3 records"


def test_every_status_has_at_least_1():
    counts = Counter(r["status"] for r in LOAN_APPLICATIONS)
    for status in STATUSES:
        assert counts.get(status, 0) >= 1, f"{status} has no records"


def test_fraud_flag_percentage_in_band():
    n = len(LOAN_APPLICATIONS)
    flagged = sum(1 for r in LOAN_APPLICATIONS if r["flagged_for_fraud_review"])
    pct = 100.0 * flagged / n
    assert 10.0 <= pct <= 30.0, f"fraud-flag rate {pct:.1f}% is outside [10%, 30%]"


def test_days_since_created_in_range():
    for r in LOAN_APPLICATIONS:
        assert 0 <= r["days_since_created"] <= 30


def test_record_ids_unique():
    ids = [r["record_id"] for r in LOAN_APPLICATIONS]
    assert len(ids) == len(set(ids))
