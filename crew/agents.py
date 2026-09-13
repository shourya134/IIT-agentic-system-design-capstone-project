"""CrewAI agent definitions (Part 2, Task 7).

Least-autonomy note (Part 4, Task 15): `check_loan_application_status` is
wired into `tools=[...]` ONLY inside build_lookup_agent(). No other
build_*_agent() function in this module references LoanLookupTool at all --
the restriction is enforced by simply never wiring the tool anywhere else,
not by a runtime check. governance/governance.py verifies this by identity
inspection across every agent this module builds.
"""
import config  # noqa: F401  (must be imported before crewai)
from crewai import Agent

from llm.crewai_mock_llm import MockLLM
from tools.loan_lookup import LoanLookupTool
from tools.rag_tool import RagLookupTool


def build_retrieval_agent() -> Agent:
    return Agent(
        role="Retrieval Agent",
        goal="Answer loan and account policy questions using only the knowledge base.",
        backstory=(
            "A support specialist who only ever answers from the official "
            "policy knowledge base and clearly says so when a question is "
            "out of scope."
        ),
        llm=MockLLM("retrieval_agent"),
        tools=[RagLookupTool()],
        verbose=False,
    )


def build_lookup_agent() -> Agent:
    return Agent(
        role="Lookup Agent",
        goal="Look up a specific loan application's status and escalation score.",
        backstory=(
            "A loan-operations specialist who is the ONLY member of this "
            "crew authorized to query the loan-application system."
        ),
        llm=MockLLM("lookup_agent"),
        tools=[LoanLookupTool()],
        verbose=False,
    )


def build_composer_agent() -> Agent:
    return Agent(
        role="Response Composer",
        goal="Combine the retrieval and lookup findings into one structured final answer.",
        backstory=(
            "An editor who merges specialist findings into a single, "
            "well-formed JSON response for the end user."
        ),
        llm=MockLLM("composer"),
        tools=[],
        verbose=False,
    )


def build_all_agents() -> dict[str, Agent]:
    return {
        "retrieval_agent": build_retrieval_agent(),
        "lookup_agent": build_lookup_agent(),
        "composer_agent": build_composer_agent(),
    }
