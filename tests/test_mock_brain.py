"""Framework-free regression tests for the two spec-called-out pitfalls.
These import nothing from crewai/autogen_agentchat/langchain_core, so they
pass regardless of whether those heavier packages are installed correctly.
"""
import inspect

from llm.mock_llm import MockBrain, ToolSpec


def test_schema_dispatch_not_name_dispatch():
    """The RAG tool is named 'rag_lookup' (contains the substring 'lookup').
    A name-substring dispatcher would misroute a record-id query to it.
    Schema-based dispatch (arg_fields) must route correctly regardless of
    tool ordering or naming."""
    brain = MockBrain()
    rag_tool_spec = ToolSpec(name="rag_lookup", arg_fields=frozenset({"query"}))
    lookup_tool_spec = ToolSpec(name="check_loan_application_status", arg_fields=frozenset({"record_id"}))

    # rag_tool_spec listed FIRST and its name contains "lookup" -- a naive
    # `"lookup" in tool.name` check would pick this one.
    decision = brain.decide("lookup_agent", "What is the status of LN0001?", [rag_tool_spec, lookup_tool_spec])

    assert decision.kind == "tool_call"
    assert decision.tool_invocation.tool_name == "check_loan_application_status"
    assert decision.tool_invocation.arguments == {"record_id": "LN0001"}


def test_query_dispatch_when_no_record_id():
    brain = MockBrain()
    rag_tool_spec = ToolSpec(name="rag_lookup", arg_fields=frozenset({"query"}))
    decision = brain.decide("retrieval_agent", "What documents do I need for KYC?", [rag_tool_spec])

    assert decision.kind == "tool_call"
    assert decision.tool_invocation.tool_name == "rag_lookup"
    assert decision.tool_invocation.arguments == {"query": "What documents do I need for KYC?"}


def test_no_observation_string_scanning_in_source():
    """Structural guard: mock_llm.py must never scan text for the literal
    marker 'Observation:' -- CrewAI's built-in ReAct system-prompt template
    contains that exact string as example text, and a parser hunting for it
    would misfire on the template before any tool ever runs."""
    source = inspect.getsource(MockBrain)
    assert "Observation:" not in source


def test_composer_role_has_no_tools_and_returns_final_answer():
    brain = MockBrain()
    decision = brain.decide("composer", "some upstream context", [])
    assert decision.kind == "final_answer"
    assert decision.final_text is not None
