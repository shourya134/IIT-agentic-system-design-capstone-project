"""LangChain session memory demo (Part 2, Task 8).

Uses InMemoryChatMessageHistory + RunnableWithMessageHistory per the
assignment's explicit instruction. RunnableWithMessageHistory raises a
LangChainDeprecationWarning pointing at LangGraph's persistence layer -- this
is expected (config.py silences it for clean transcripts) and the class
still functions correctly; we deliberately do NOT migrate to LangGraph since
that would contradict the assignment.

Memory here is in-process only (a dict of session_id -> history) and does
not need to survive a restart, per the spec.
"""
import config  # noqa: F401
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from llm.langchain_mock_llm import MockChatModel

_STORE: dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in _STORE:
        _STORE[session_id] = InMemoryChatMessageHistory()
    return _STORE[session_id]


prompt = ChatPromptTemplate.from_messages(
    [MessagesPlaceholder(variable_name="history"), ("human", "{input}")]
)
chain = prompt | MockChatModel()

with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)


def _ask(session_id: str, text: str) -> str:
    result = with_history.invoke({"input": text}, config={"configurable": {"session_id": session_id}})
    return result.content


def demo_multiturn(session_id: str = "demo-session") -> list[str]:
    """Multi-turn transcript proving state is carried WITHIN one session."""
    lines = []
    turns = [
        "My record id is LN0007.",
        "What is my record id?",
    ]
    for turn in turns:
        reply = _ask(session_id, turn)
        lines.append(f"User: {turn}")
        lines.append(f"Assistant: {reply}")
    return lines


def demo_fresh_session(session_id: str = "fresh-session") -> list[str]:
    """SEPARATE fresh-conversation transcript proving state is correctly
    absent/reset for a brand-new session_id (no leakage from demo_multiturn)."""
    lines = []
    turn = "What is my record id?"
    reply = _ask(session_id, turn)
    lines.append(f"User: {turn}")
    lines.append(f"Assistant: {reply}")
    return lines


if __name__ == "__main__":
    print("=== Multi-turn (state carried) ===")
    for line in demo_multiturn():
        print(line)
    print("=== Fresh session (state absent) ===")
    for line in demo_fresh_session():
        print(line)
