import sys
import statistics
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: F401,E402
from data.dataset import LOAN_APPLICATIONS  # noqa: E402
from tools.loan_lookup import ESCALATION_THRESHOLD, check_loan_application_status  # noqa: E402
from scripts._demo_utils import Logger  # noqa: E402


def main():
    log = Logger()
    log("Part 2, Task 6: check_loan_application_status + designed escalation_score")
    log(
        "Formula: escalation_score = 0.7 * flagged_for_fraud_review + "
        "0.3 * (days_since_created / 30)"
    )
    log(f"Escalation threshold: {ESCALATION_THRESHOLD}")

    days = [r["days_since_created"] for r in LOAN_APPLICATIONS]
    p80 = statistics.quantiles(days, n=5)[3]
    log(f"80th percentile of days_since_created in the generated dataset: {p80:.1f}")
    log(
        "Justification: any fraud-flagged record scores >= 0.7 (always "
        "escalates); any unflagged record scores <= 0.3 (never escalates on "
        "age alone) -- 0.5 sits cleanly in the gap between those two clusters."
    )

    flagged = [r for r in LOAN_APPLICATIONS if r["flagged_for_fraud_review"]][:2]
    unflagged = [r for r in LOAN_APPLICATIONS if not r["flagged_for_fraud_review"]][:2]
    log("Sample lookups (flagged records):")
    for r in flagged:
        log(f"  {check_loan_application_status(r['record_id'])}")
    log("Sample lookups (unflagged records):")
    for r in unflagged:
        log(f"  {check_loan_application_status(r['record_id'])}")
    log("Lookup for a non-existent record:")
    log(f"  {check_loan_application_status('LN9999')}")

    log.save("part2_task6.txt")


if __name__ == "__main__":
    main()
