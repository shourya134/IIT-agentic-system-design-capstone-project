"""Autogen 2-agent RoundRobinGroupChat review stage (Part 4, Task 14).

*** VERIFY AGAINST YOUR INSTALLED autogen-agentchat VERSION *** -- the
constructor arguments below (`termination_condition`, `custom_message_types`,
`output_content_type`) match autogen-agentchat's documented 0.4+ AgentChat
API. Two spec-called-out gotchas this file is built around:
  1. `output_content_type=VerdictModel` on the Final-Editor AssistantAgent
     requires the Team itself to be constructed with
     `custom_message_types=[StructuredMessage[VerdictModel]]`, or the run
     crashes with `ValueError: Message type ... is not registered`.
  2. `MaxMessageTermination` counts the INITIATING TASK MESSAGE as message
     #1. So `MaxMessageTermination(3)` (not `(2)`) is what lets both the
     reviewer (#2) and the editor (#3) each speak exactly once.
"""
import asyncio

import config  # noqa: F401
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.messages import StructuredMessage
from autogen_agentchat.teams import RoundRobinGroupChat

from llm.autogen_mock_client import MockChatCompletionClient
from review.verdict_schema import VerdictModel


def build_review_team() -> RoundRobinGroupChat:
    reviewer = AssistantAgent(
        "PolicyComplianceReviewer",
        model_client=MockChatCompletionClient("reviewer"),
        system_message=(
            "You review a draft support answer against the retrieved policy "
            "context and flag any claim the context does not support."
        ),
    )
    editor = AssistantAgent(
        "FinalEditor",
        model_client=MockChatCompletionClient("editor"),
        system_message="You produce the final approved-or-revised answer as a structured verdict.",
        output_content_type=VerdictModel,
    )
    return RoundRobinGroupChat(
        [reviewer, editor],
        termination_condition=MaxMessageTermination(3),
        custom_message_types=[StructuredMessage[VerdictModel]],
    )


def _extract_verdict(messages) -> VerdictModel:
    for msg in reversed(messages):
        content = getattr(msg, "content", None)
        if isinstance(content, VerdictModel):
            return content
        if isinstance(content, str):
            try:
                return VerdictModel.model_validate_json(content)
            except ValueError:
                continue
    raise ValueError("Could not extract a VerdictModel from the review team's messages.")


async def run_review(draft: str, context: str) -> VerdictModel:
    team = build_review_team()
    task = f"DRAFT:\n{draft}\n\nCONTEXT:\n{context}"
    result = await team.run(task=task)
    return _extract_verdict(result.messages)


def demo_approved_case() -> VerdictModel:
    """Draft with no unsupported claims -> reviewer approves unchanged."""
    draft = "Personal loan applicants must be salaried with a minimum monthly income of INR 25,000."
    context = (
        "Personal loan applicants must be salaried or self-employed with a "
        "minimum monthly income of INR 25,000 and at least two years of "
        "continuous work or business history."
    )
    return asyncio.run(run_review(draft, context))


def demo_revision_case() -> VerdictModel:
    """Draft with a deliberately injected ungrounded claim -> caught and revised."""
    draft = (
        "Your loan is guaranteed approval with zero interest and no "
        "documentation required."
    )
    context = "Personal loan applicants must be salaried with a minimum monthly income of INR 25,000."
    return asyncio.run(run_review(draft, context))


if __name__ == "__main__":
    print("=== Approved-unchanged case ===")
    verdict = demo_approved_case()
    print(verdict.model_dump_json(indent=2))

    print("=== Revised case (ungrounded claim injected) ===")
    verdict = demo_revision_case()
    print(verdict.model_dump_json(indent=2))
