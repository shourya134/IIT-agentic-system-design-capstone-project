"""15-query LLM-as-judge test set (Part 3, Task 13).

Covers every one of the 12 required KB topics at least once, plus 3
deliberately out-of-scope/edge-case queries (>= 2 required).
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TestQuery:
    query: str
    expected_topic: Optional[str]  # KB doc_id, or None for out-of-scope
    in_scope: bool
    keywords: list = field(default_factory=list)  # for the Completeness score


TEST_QUERIES: list[TestQuery] = [
    TestQuery(
        "What are the eligibility criteria for a home loan?",
        "loan_eligibility", True,
        ["loan-to-value", "income", "age", "eligibility"],
    ),
    TestQuery(
        "How is my monthly EMI calculated?",
        "emi_calculation", True,
        ["principal", "interest", "tenure", "emi"],
    ),
    TestQuery(
        "What fees does my credit card charge for late payment?",
        "credit_card_fees", True,
        ["fee", "late", "credit card", "charge"],
    ),
    TestQuery(
        "What documents are required for KYC verification?",
        "kyc_requirements", True,
        ["identity", "address", "kyc", "proof"],
    ),
    TestQuery(
        "How do I dispute a fraudulent transaction on my account?",
        "fraud_dispute_resolution", True,
        ["fraud", "dispute", "investigation", "reversed"],
    ),
    TestQuery(
        "What is the process to close my bank account?",
        "account_closure", True,
        ["closure", "form", "chequebook", "dues"],
    ),
    TestQuery(
        "What determines the interest rate slab for my loan?",
        "interest_rate_slabs", True,
        ["interest", "credit score", "slab", "rate"],
    ),
    TestQuery(
        "Will I be charged a penalty for prepaying my loan?",
        "prepayment_penalty", True,
        ["prepayment", "penalty", "foreclosure"],
    ),
    TestQuery(
        "What happens if I don't maintain the minimum balance in my account?",
        "minimum_balance", True,
        ["minimum balance", "charge", "average balance"],
    ),
    TestQuery(
        "What factors affect my credit score?",
        "credit_score_factors", True,
        ["payment history", "credit utilization", "credit score"],
    ),
    TestQuery(
        "Can I add another person to my joint account?",
        "joint_account_rules", True,
        ["joint account", "holder", "consent"],
    ),
    TestQuery(
        "Am I eligible to open an NRI account?",
        "nri_account_eligibility", True,
        ["nre", "nro", "nri", "passport"],
    ),
    TestQuery("What is the weather like in Tokyo today?", None, False, []),
    TestQuery("Write a haiku about the moon.", None, False, []),
    TestQuery("Who won the last FIFA World Cup?", None, False, []),
]

assert len(TEST_QUERIES) == 15
