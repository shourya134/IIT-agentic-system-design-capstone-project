"""RAG lookup CrewAI tool (Part 2, Task 7 dependency).

Named "rag_lookup" DELIBERATELY -- it contains the substring "lookup" so that
llm/mock_llm.py's tool-dispatch logic can be shown to route correctly by
inspecting the tool's declared argument schema (args_schema field names)
rather than by substring-matching tool.name, which would misclassify this
tool as the loan-lookup tool. See tests/test_mock_brain.py for the
regression test that pins this behavior.
"""
from pydantic import BaseModel, Field

import config
from rag.generation import cached_answer_or_fallback
from rag.vector_store import FIXED_COLLECTION

# The collection Part 1 Task 5 recommends deploying (measured: kb_fixed avg
# precision 0.900 vs kb_sentence 0.433, equal recall -- see
# transcripts/part1_task5.txt and README.md). Kept as a module-level constant
# so crew_setup.py and the demo scripts agree on one fixed input.
RECOMMENDED_COLLECTION = FIXED_COLLECTION

# Incremented on every real call, so demo scripts can prove this specific
# tool fired (as opposed to just asserting the crew produced some output).
CALL_COUNT = 0


def rag_lookup(query: str) -> str:
    global CALL_COUNT
    CALL_COUNT += 1
    answer, is_fallback, cache_hit = cached_answer_or_fallback(query, RECOMMENDED_COLLECTION)
    return answer


class RagQueryArgs(BaseModel):
    query: str = Field(..., description="The user's policy/knowledge-base question")


try:
    from crewai.tools import BaseTool

    class RagLookupTool(BaseTool):
        name: str = "rag_lookup"
        description: str = (
            "Answer a loan/account policy question using only the retrieved "
            "knowledge-base context. Returns an 'I don't know' fallback for "
            "out-of-scope questions."
        )
        args_schema: type[BaseModel] = RagQueryArgs

        def _run(self, query: str) -> str:
            return rag_lookup(query)

except ImportError:  # pragma: no cover
    RagLookupTool = None


if __name__ == "__main__":
    print(rag_lookup("What documents do I need for KYC verification?"))
    print(rag_lookup("What is the weather like today?"))
