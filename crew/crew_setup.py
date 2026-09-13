"""Wires the 3-agent crew together and runs it via .kickoff() (Part 2, Task 7)."""
import config  # noqa: F401  (must be imported before crewai)
from crewai import Crew, Process, Task

from crew.agents import build_all_agents
from crew.output_schema import ResponseSchema, parse_crew_output


def build_crew(user_query: str) -> Crew:
    agents = build_all_agents()

    retrieval_task = Task(
        description=user_query,
        expected_output="A grounded answer to the policy question, or an 'I don't know' fallback if out of scope.",
        agent=agents["retrieval_agent"],
    )
    lookup_task = Task(
        description=user_query,
        expected_output="The loan application's status, amount, and escalation score, or a not-found message.",
        agent=agents["lookup_agent"],
    )
    composer_task = Task(
        description=user_query,
        expected_output="A JSON object with fields: answer, sources, confidence.",
        agent=agents["composer_agent"],
        context=[retrieval_task, lookup_task],
    )

    return Crew(
        agents=[agents["retrieval_agent"], agents["lookup_agent"], agents["composer_agent"]],
        tasks=[retrieval_task, lookup_task, composer_task],
        process=Process.sequential,
        verbose=False,
    )


def run_crew(user_query: str) -> ResponseSchema:
    crew = build_crew(user_query)
    result = crew.kickoff()
    raw = result.raw if hasattr(result, "raw") else str(result)
    return parse_crew_output(raw)


if __name__ == "__main__":
    # Query with no record ID -> only the RAG tool should fire meaningfully.
    print("--- RAG-only query ---")
    print(run_crew("What documents do I need for KYC verification?"))

    # Query with a record ID -> the lookup tool should fire.
    print("--- Lookup query ---")
    print(run_crew("What is the status of application LN0001?"))
