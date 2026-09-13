"""Application-layer governance: least-autonomy enforcement and risk
classification (Part 4, Task 15).

Least-autonomy enforcement design: `check_loan_application_status` (wrapped
as LoanLookupTool) is referenced in exactly one place in the whole codebase --
crew/agents.py::build_lookup_agent()'s `tools=[...]` list. No other
build_*_agent() function imports or wires LoanLookupTool at all. This is a
structural guard (the tool is simply never given to any other agent), not a
runtime permission check -- there is nothing to "bypass" because no other
agent object ever holds a reference to the tool. assert_tool_ownership below
verifies this by identity inspection across every agent the crew builds, so
a regression (e.g. someone later wiring LoanLookupTool into the retrieval or
composer agent) would be caught immediately.
"""
import config  # noqa: F401
from tools.loan_lookup import LoanLookupTool

GOVERNANCE_LAYERS = {
    "Data": "PII masking at ingestion and before any logging (crew/guardrails.py + api/logging_utils.py).",
    "Application": "Least-autonomy tool scoping: only the Lookup Agent holds check_loan_application_status.",
    "Runtime": "Per-request token/cost budget cap (governance/budget.py); guardrail refusals before crew execution.",
    "Compliance": "Autogen review stage (review/autogen_review.py) plus an append-only JSONL audit log with trace IDs.",
}


def assert_tool_ownership() -> list[str]:
    from crew.agents import build_all_agents

    agents = build_all_agents()
    lines = []
    owners = []
    for agent_key, agent in agents.items():
        has_lookup_tool = any(isinstance(t, LoanLookupTool) for t in (agent.tools or []))
        if has_lookup_tool:
            owners.append(agent_key)
        status = "HAS lookup tool" if has_lookup_tool else "does not have lookup tool"
        lines.append(f"  {agent_key}: {status}")

    ok = owners == ["lookup_agent"]
    lines.append(f"Least-autonomy check: {'PASS' if ok else 'FAIL'} (owners={owners})")
    if not ok:
        raise AssertionError(f"Least-autonomy violation: expected only lookup_agent to hold the tool, got {owners}")
    return lines


def classify_risk() -> dict:
    return {
        "level": "Medium-High",
        "justification": (
            "This system handles fixed-format PII (PAN/Aadhaar/bank account numbers) "
            "and financial data (loan status, loan amount, a fraud-review flag and "
            "escalation score) about real-seeming member records, which places it "
            "above Low risk (simple summarization/transcription) and into the "
            "Medium/High financial-data band described in the scenario's risk scheme. "
            "It stops short of the most severe High-risk examples (e.g. medical data "
            "or automated hiring/credit decisions) because under MOCK_LLM it makes no "
            "real money-movement, autonomous approval/denial, or external API calls -- "
            "it only surfaces existing record status and policy text for a human "
            "support agent to act on. We therefore classify it Medium-High: strict "
            "PII-masking, logging, and human-in-the-loop review (the Autogen stage) "
            "are required, matching the caution owed to financial-data systems."
        ),
    }


if __name__ == "__main__":
    print("=== Least-autonomy enforcement ===")
    for line in assert_tool_ownership():
        print(line)
    print("=== Risk classification ===")
    risk = classify_risk()
    print(f"Level: {risk['level']}")
    print(f"Justification: {risk['justification']}")
    print("=== Governance layers ===")
    for layer, desc in GOVERNANCE_LAYERS.items():
        print(f"  {layer}: {desc}")
