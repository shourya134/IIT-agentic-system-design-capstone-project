"""Loan application status lookup tool with a designed escalation score
(Part 2, Task 6).

escalation_score formula:
    escalation_score = 0.7 * float(flagged_for_fraud_review) + 0.3 * (days_since_created / 30)

Rationale: fraud-review flagging is the dominant signal (weight 0.7) because
it is an explicit human/system judgement that something is suspicious.
Recency (days_since_created, normalized to [0, 1] over the 0-30 day window
used by the dataset generator) is a secondary signal (weight 0.3) -- an
application that has sat unresolved for a long time is itself a mild risk
signal even without a fraud flag, but should never on its own be as
disqualifying as an actual fraud flag.

Threshold justification (recomputed against the actual generated dataset by
scripts/demo_part2_task6_lookup.py, not hand-picked in isolation):
  - Any flagged_for_fraud_review=True record scores >= 0.7, comfortably
    above 0.5 -- always escalates regardless of age.
  - Any flagged_for_fraud_review=False record scores <= 0.3 (0.3 * 30/30),
    comfortably below 0.5 -- age alone never triggers escalation.
  - 0.5 therefore falls in the empty gap between the two clusters the
    formula produces, so it is a clean separating threshold rather than an
    arbitrary pick. This is also close to the 80th percentile of
    days_since_created for a uniform 0-30 sample (~24 days out of a 0-30
    range, i.e. the top quintile by age), which is the recency band the
    formula's 0.3 weight is calibrated to flag as "notably old" IF combined
    with a fraud flag.
"""
from pydantic import BaseModel, Field

import config
from data.dataset import LOAN_APPLICATIONS, get_record

ESCALATION_THRESHOLD = config.ESCALATION_THRESHOLD

# Incremented on every real call, so demo scripts can prove this specific
# tool fired (as opposed to just asserting the crew produced some output).
CALL_COUNT = 0


def check_loan_application_status(record_id: str) -> dict:
    global CALL_COUNT
    CALL_COUNT += 1
    record = get_record(record_id, LOAN_APPLICATIONS)
    if record is None:
        return {
            "record_id": record_id,
            "found": False,
            "error": f"No application found with record_id={record_id!r}",
        }

    recency_signal = min(record["days_since_created"], 30) / 30.0
    escalation_score = 0.7 * float(record["flagged_for_fraud_review"]) + 0.3 * recency_signal
    escalation_score = round(escalation_score, 4)

    return {
        "record_id": record["record_id"],
        "found": True,
        "status": record["status"],
        "loan_amount_inr": record["loan_amount_inr"],
        "escalation_score": escalation_score,
        "escalation_recommended": escalation_score >= ESCALATION_THRESHOLD,
    }


class LoanLookupArgs(BaseModel):
    record_id: str = Field(..., description="The loan application record ID to look up, e.g. LN0001")


try:
    from crewai.tools import BaseTool

    class LoanLookupTool(BaseTool):
        name: str = "check_loan_application_status"
        description: str = (
            "Look up a loan application's status, loan amount, and escalation "
            "score by its record_id (e.g. LN0001)."
        )
        args_schema: type[BaseModel] = LoanLookupArgs

        def _run(self, record_id: str) -> dict:
            return check_loan_application_status(record_id)

except ImportError:  # pragma: no cover - allows dataset/tool logic to be
    # unit-tested without crewai installed.
    LoanLookupTool = None


if __name__ == "__main__":
    import statistics

    days = [r["days_since_created"] for r in LOAN_APPLICATIONS]
    p80 = statistics.quantiles(days, n=5)[3]  # 80th percentile
    print(f"80th percentile of days_since_created in generated dataset: {p80:.1f}")

    flagged = [r for r in LOAN_APPLICATIONS if r["flagged_for_fraud_review"]][:2]
    unflagged = [r for r in LOAN_APPLICATIONS if not r["flagged_for_fraud_review"]][:2]
    for r in flagged + unflagged:
        print(check_loan_application_status(r["record_id"]))
    print(check_loan_application_status("LN9999"))
