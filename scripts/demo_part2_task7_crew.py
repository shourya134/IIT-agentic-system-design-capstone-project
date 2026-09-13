import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: F401,E402
import tools.loan_lookup as loan_lookup  # noqa: E402
import tools.rag_tool as rag_tool  # noqa: E402
from crew.crew_setup import run_crew  # noqa: E402
from scripts._demo_utils import Logger  # noqa: E402


def main():
    log = Logger()
    log("Part 2, Task 7: CrewAI crew, 3 agents, .kickoff() demonstrated on two distinct queries")

    rag_calls_before, lookup_calls_before = rag_tool.CALL_COUNT, loan_lookup.CALL_COUNT
    log("--- Query 1 (no record ID -> should invoke the RAG tool) ---")
    query1 = "What documents do I need for KYC verification?"
    response1 = run_crew(query1)
    log(f"Query: {query1}")
    log(f"Response: {response1.model_dump_json()}")
    log(f"rag_tool.CALL_COUNT delta: {rag_tool.CALL_COUNT - rag_calls_before}")
    log(f"loan_lookup.CALL_COUNT delta: {loan_lookup.CALL_COUNT - lookup_calls_before}")
    assert rag_tool.CALL_COUNT > rag_calls_before, "RAG tool should have fired for query 1"

    rag_calls_before, lookup_calls_before = rag_tool.CALL_COUNT, loan_lookup.CALL_COUNT
    log("--- Query 2 (contains a record ID -> should invoke the lookup tool) ---")
    query2 = "What is the status of application LN0001?"
    response2 = run_crew(query2)
    log(f"Query: {query2}")
    log(f"Response: {response2.model_dump_json()}")
    log(f"rag_tool.CALL_COUNT delta: {rag_tool.CALL_COUNT - rag_calls_before}")
    log(f"loan_lookup.CALL_COUNT delta: {loan_lookup.CALL_COUNT - lookup_calls_before}")
    assert loan_lookup.CALL_COUNT > lookup_calls_before, "Lookup tool should have fired for query 2"

    log("PASS: RAG tool and lookup tool each demonstrably fired on different queries.")
    log.save("part2_task7.txt")


if __name__ == "__main__":
    main()
