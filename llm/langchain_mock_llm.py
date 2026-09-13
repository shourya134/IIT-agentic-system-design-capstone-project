"""LangChain BaseChatModel adapter around MockBrain (Part 2, Task 8).

*** VERIFY AGAINST YOUR INSTALLED langchain-core VERSION BEFORE TRUSTING THIS
FILE *** -- `_generate`'s signature and `ChatResult`/`ChatGeneration`
construction are based on langchain-core's documented custom chat model
guide; confirm against `langchain_core.language_models.chat_models
.BaseChatModel` in your installed version if anything fails to import.

Unlike the CrewAI adapter, there are no tools here -- this is a pure
history-aware chat used to demonstrate LangChain session memory
(InMemoryChatMessageHistory + RunnableWithMessageHistory). The full message
list `RunnableWithMessageHistory` assembles (prior turns + the current human
turn) is exactly what's needed here, since the memory demo's entire point is
to prove prior turns are visible -- this is the one adapter that
legitimately reads more than just the latest message.
"""
from typing import Any, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from llm.mock_llm import MockBrain


class MockChatModel(BaseChatModel):
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        brain = MockBrain()
        history_texts = [m.content for m in messages[:-1]]
        current_message = messages[-1].content if messages else ""
        reply_text = brain.chat_memory_reply(history_texts, current_message)
        message = AIMessage(content=reply_text)
        return ChatResult(generations=[ChatGeneration(message=message)])

    @property
    def _llm_type(self) -> str:
        return "mock-chat-model"
