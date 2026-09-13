"""MockBrain: the single shared, framework-free decision core behind every
MOCK_LLM adapter (CrewAI, LangChain, Autogen, the Task 13 judge).

This module deliberately imports NOTHING from crewai/autogen_agentchat/
langchain_core so it can be unit-tested in complete isolation (see
tests/test_mock_brain.py) and so the two pitfalls called out in the
assignment spec are fixed here, once, structurally:

Pitfall A (the "Observation:" trap): CrewAI's built-in ReAct system-prompt
template contains the literal example text "Observation: the result of the
action". A parser that scans conversation/prompt text for the substring
"Observation:" to find a tool's result will match this template text before
any tool has run. MockBrain never scans prompt or system-prompt text for
markers like this -- `decide()` only ever looks at the CALLER-SUPPLIED
`task_text` (which adapters must populate from the model's own latest
generated/human turn only, never the system prompt), and "has a tool already
run" is tracked by the adapter as a plain Python variable/control-flow state,
never inferred from scanning text for a keyword.

Pitfall B (dispatch-by-name trap): tool dispatch is keyed off each tool's
declared ARGUMENT-SCHEMA field names (a ToolSpec's `arg_fields`), never off
`tool.name`. tools/rag_tool.py is deliberately named "rag_lookup" (containing
the substring "lookup") specifically to prove this: a name-substring check
would misroute it to the loan-lookup path, but schema-based dispatch (this
tool takes a `query` field, not a `record_id` field) always resolves it
correctly. See tests/test_mock_brain.py::test_schema_dispatch_not_name_dispatch.
"""
import re
from dataclasses import dataclass
from typing import Any, Literal, Optional

RECORD_ID_PATTERN = re.compile(r"\b[A-Z]{2}\d{4}\b")


@dataclass(frozen=True)
class ToolSpec:
    name: str
    arg_fields: frozenset


@dataclass(frozen=True)
class ToolInvocation:
    tool_name: str
    arguments: dict


@dataclass
class MockDecision:
    kind: Literal["tool_call", "final_answer"]
    tool_invocation: Optional[ToolInvocation] = None
    final_text: Optional[str] = None


class MockBrain:
    """Deterministic decision logic shared by every framework adapter."""

    def decide(self, agent_role: str, task_text: str, tool_specs: list[ToolSpec]) -> MockDecision:
        record_tool = next((t for t in tool_specs if t.arg_fields == frozenset({"record_id"})), None)
        query_tool = next((t for t in tool_specs if t.arg_fields == frozenset({"query"})), None)

        match = RECORD_ID_PATTERN.search(task_text)
        if record_tool is not None and match is not None:
            return MockDecision(
                kind="tool_call",
                tool_invocation=ToolInvocation(record_tool.name, {"record_id": match.group()}),
            )
        if query_tool is not None:
            return MockDecision(
                kind="tool_call",
                tool_invocation=ToolInvocation(query_tool.name, {"query": task_text.strip()}),
            )
        return MockDecision(kind="final_answer", final_text=self.compose_final_answer(agent_role, task_text, None, None))

    def compose_final_answer(self, agent_role: str, task_text: str, tool_name: Optional[str], tool_result: Any) -> str:
        if agent_role == "retrieval_agent":
            # tool_result is already the RAG tool's grounded (or fallback) text.
            return str(tool_result) if tool_result is not None else "No policy context was retrieved."

        if agent_role == "lookup_agent":
            if not isinstance(tool_result, dict):
                return "No loan application record_id was found in the request."
            if not tool_result.get("found"):
                return tool_result.get("error", "Application not found.")
            return (
                f"Application {tool_result['record_id']} has status "
                f"'{tool_result['status']}' with loan amount INR "
                f"{tool_result['loan_amount_inr']:,}. Escalation score: "
                f"{tool_result['escalation_score']} "
                f"({'escalation recommended' if tool_result['escalation_recommended'] else 'no escalation needed'})."
            )

        if agent_role == "composer":
            return self._compose_response_json(task_text)

        return task_text

    def _compose_response_json(self, task_text: str) -> str:
        """Builds the Composer's structured JSON output (Task 9) from the
        upstream retrieval/lookup outputs CrewAI has chained into task_text
        via task `context=[...]`."""
        import json

        sources = []
        for m in re.finditer(r"Source\(s\):\s*([\w,\s]+)", task_text):
            sources.extend(s.strip() for s in m.group(1).split(",") if s.strip())

        confidence = 0.3 if "don't have enough information" in task_text else 0.85
        answer_text = task_text.strip()
        return json.dumps({"answer": answer_text, "sources": sorted(set(sources)), "confidence": confidence})

    # -- Task 8: LangChain session-memory chat (no tools, history-aware) --

    RECORD_ID_STATEMENT_PATTERN = re.compile(r"my record (?:id|ID) is\s+([A-Z]{2}\d{4})", re.IGNORECASE)

    def chat_memory_reply(self, history_texts: list[str], current_message: str) -> str:
        """Deterministic reply for the Task 8 memory demo: if the user is
        stating a record id, acknowledge it; if asking to recall it, look
        back through history_texts (prior turns in THIS session only)."""
        stated = self.RECORD_ID_STATEMENT_PATTERN.search(current_message)
        if stated:
            return f"Got it, I've noted your record ID as {stated.group(1)} for this conversation."

        if re.search(r"what.*(my )?record (?:id|ID)", current_message, re.IGNORECASE):
            for past in reversed(history_texts):
                m = self.RECORD_ID_STATEMENT_PATTERN.search(past)
                if m:
                    return f"Your record ID for this conversation is {m.group(1)}."
            return "I don't have a record ID on file yet for this conversation -- please share one."

        return "I can help with loan policy questions or application status. What would you like to know?"

    # -- Task 14: Autogen review stage (reviewer / editor) --

    UNGROUNDED_MARKERS = [
        "guaranteed approval",
        "no documentation required",
        "zero interest",
        "unlimited loan amount",
    ]

    def review_draft(self, draft: str, context: str) -> dict:
        """Reviewer role: flags any deliberately-plantable ungrounded-claim
        marker phrase present in the draft but absent from the supplied
        context. Returns {"approved": bool, "reason": str}."""
        lowered_draft = draft.lower()
        lowered_context = context.lower()
        found = [m for m in self.UNGROUNDED_MARKERS if m in lowered_draft and m not in lowered_context]
        if found:
            return {
                "approved": False,
                "reason": f"Draft contains claim(s) not supported by retrieved context: {', '.join(found)}.",
            }
        return {"approved": True, "reason": "Draft is consistent with the retrieved context."}

    def edit_verdict(self, draft: str, context: str, review_result: dict) -> dict:
        """Editor role: emits the final verdict. If approved, passes the
        draft through unchanged; otherwise strips the offending phrases and
        explains the revision."""
        if review_result["approved"]:
            return {"approved": True, "final_answer": draft, "reason": review_result["reason"]}

        revised = draft
        for marker in self.UNGROUNDED_MARKERS:
            pattern = re.compile(re.escape(marker), re.IGNORECASE)
            revised = pattern.sub("[claim removed: not supported by retrieved context]", revised)
        return {
            "approved": False,
            "final_answer": revised,
            "reason": f"Revised to remove unsupported claim(s). {review_result['reason']}",
        }
