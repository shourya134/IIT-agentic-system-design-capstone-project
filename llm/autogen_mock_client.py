"""Autogen ChatCompletionClient adapter around MockBrain (Part 4, Task 14).

*** VERIFY AGAINST YOUR INSTALLED autogen-core VERSION BEFORE TRUSTING THIS
FILE *** -- the `ChatCompletionClient` abstract base class in
`autogen_core.models` has a specific abstract-method list (create,
create_stream, actual_usage, total_usage, count_tokens, remaining_tokens,
plus `capabilities`/`model_info` properties) and this was written against
autogen-core's documented shape as of this writing. Run:
    python -c "from autogen_core.models import ChatCompletionClient; import inspect; print(inspect.getsource(ChatCompletionClient))"
and reconcile any missing/renamed abstract methods before relying on this in
review/autogen_review.py.

Design: two roles.
  - "reviewer": looks at the conversation so far (the draft answer + the
    original retrieved context, both passed in as plain message content by
    review/autogen_review.py) and calls MockBrain.review_draft(...), replying
    with free-text content describing its verdict.
  - "editor": constructed with output_content_type=VerdictModel on the
    AssistantAgent side; when `json_output` names a Pydantic model, this
    client returns CreateResult.content as a JSON string that validates
    against it, built from MockBrain.edit_verdict(...).
"""
import json
from typing import Any, AsyncGenerator, Literal, Optional, Sequence, Union

from autogen_core.models import (
    ChatCompletionClient,
    CreateResult,
    LLMMessage,
    ModelCapabilities,
    ModelInfo,
    RequestUsage,
)

from llm.mock_llm import MockBrain

_ZERO_USAGE = RequestUsage(prompt_tokens=0, completion_tokens=0)


def _extract_task_text(messages: Sequence[LLMMessage]) -> str:
    """Find the original "DRAFT:...CONTEXT:..." task message.

    By the editor's turn, `messages` also contains the reviewer's own reply,
    which echoes a second DRAFT/CONTEXT block back into its own content (see
    the "reviewer" branch below) for logging. Naively concatenating every
    message's content and partitioning on the first "CONTEXT:" marker would
    swallow that trailing echo into "context" -- including the very phrases
    the reviewer just flagged as unsupported -- silently overturning a
    correct "approved=False" verdict. The original task message is the only
    one whose content starts with "DRAFT:" (the reviewer's reply starts with
    "APPROVED="), so picking it out directly sidesteps that instead of
    patching the parse.
    """
    for m in messages:
        content = getattr(m, "content", "")
        if isinstance(content, str) and content.startswith("DRAFT:"):
            return content
    return ""


class MockChatCompletionClient(ChatCompletionClient):
    def __init__(self, agent_role: Literal["reviewer", "editor"]):
        self.agent_role = agent_role
        self.brain = MockBrain()
        self._total_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)

    async def create(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Any] = (),
        tool_choice: Any = "auto",
        json_output: Optional[Any] = None,
        extra_create_args: Optional[Any] = None,
        cancellation_token: Optional[Any] = None,
    ) -> CreateResult:
        task_text = _extract_task_text(messages)
        # review/autogen_review.py formats the task message as:
        #   "DRAFT:\n<draft>\n\nCONTEXT:\n<context>"
        draft, context = _split_draft_context(task_text)

        if self.agent_role == "reviewer":
            review_result = self.brain.review_draft(draft, context)
            content = (
                f"APPROVED={review_result['approved']} | {review_result['reason']}\n"
                f"DRAFT:\n{draft}\n\nCONTEXT:\n{context}"
            )
            return CreateResult(finish_reason="stop", content=content, usage=_ZERO_USAGE, cached=False)

        # agent_role == "editor"
        review_result = self.brain.review_draft(draft, context)
        verdict_dict = self.brain.edit_verdict(draft, context, review_result)

        if json_output is not None:
            content = json.dumps(verdict_dict)
        else:
            content = verdict_dict["final_answer"]
        return CreateResult(finish_reason="stop", content=content, usage=_ZERO_USAGE, cached=False)

    async def create_stream(
        self, messages, *, tools=(), tool_choice="auto", json_output=None,
        extra_create_args=None, cancellation_token=None,
    ) -> AsyncGenerator[Union[str, CreateResult], None]:
        result = await self.create(
            messages, tools=tools, tool_choice=tool_choice, json_output=json_output,
            extra_create_args=extra_create_args, cancellation_token=cancellation_token,
        )
        yield str(result.content)
        yield result

    async def close(self) -> None:
        pass

    def actual_usage(self) -> RequestUsage:
        return _ZERO_USAGE

    def total_usage(self) -> RequestUsage:
        return self._total_usage

    def count_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Any] = ()) -> int:
        return len(_extract_text(messages)) // 4

    def remaining_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Any] = ()) -> int:
        return 8192 - self.count_tokens(messages, tools=tools)

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(vision=False, function_calling=False, json_output=True)

    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(
            vision=False, function_calling=False, json_output=True, family="mock", structured_output=True
        )


def _split_draft_context(text: str) -> tuple[str, str]:
    _, _, after_draft = text.partition("DRAFT:")
    draft, _, context_part = after_draft.partition("CONTEXT:")
    return draft.strip(), context_part.strip()
