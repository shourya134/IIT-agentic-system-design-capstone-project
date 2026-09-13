"""CrewAI BaseLLM adapter around the shared MockBrain.

*** VERIFY AGAINST YOUR INSTALLED crewai VERSION BEFORE TRUSTING THIS FILE ***
This subclasses `crewai.llms.base_llm.BaseLLM`, CrewAI's documented extension
point for a non-litellm LLM. The abstract method list and the exact shapes of
`messages` / `tools` / `available_functions` passed into `call()` are based on
CrewAI's published custom-LLM guide as of this writing, but CrewAI moves
quickly -- open `site-packages/crewai/llms/base_llm.py` in your installed
version and confirm: (1) the abstract methods below match, (2) `tools` really
is a list of OpenAI-function-schema dicts (`tool["function"]["parameters"]
["properties"]`), (3) `available_functions` really is a `name -> callable`
dict, and (4) `Agent(llm=<BaseLLM instance>)` is accepted directly. Adjust
this file's plumbing (not llm/mock_llm.py's decision logic) if anything
differs.

Design note (why this avoids both spec-called-out pitfalls structurally):
- `task_text` -- the text `MockBrain` actually reasons over -- is always
  either `messages[-1]["content"]` (first turn) or the original task's own
  `"user"`-role message recovered from history (post-tool-call turn, see
  `_first_user_content`) -- never the full raw messages list, so CrewAI's
  system-prompt ReAct template text (which contains the literal example
  "Observation: the result of the action") can never be scanned or misread
  as a real tool observation (Pitfall A).
- Whether a tool has run is tracked by this method actually having called
  `available_functions[...]` (or, under native tool calling -- see below --
  by CrewAI itself having appended a real `"tool"`-role result message) and
  holding the real Python return value in `tool_result` -- never inferred by
  pattern-matching prompt text.
- Tool dispatch happens inside MockBrain.decide() by schema shape, not name
  substring matching (Pitfall B) -- see llm/mock_llm.py.
- Native tool-calling protocol: the installed crewai's default `AgentExecutor`
  always calls `.call()` with `available_functions=None` and expects the LLM
  itself to return either a final-answer string or a list of OpenAI-shaped
  tool-call request dicts (`[{"id":..., "function": {"name":..., "arguments":
  "<json>"}}]`); CrewAI then executes the real tool and calls `.call()` again
  with the result appended as a `"tool"`-role message. So `call()` no longer
  "owns" the full decide -> execute -> compose loop in one invocation the way
  earlier crewai versions' custom-LLM guide documented -- it spans two calls:
  first return a tool-call request, then (detected via the trailing `"tool"`
  message) compose the final answer from CrewAI's real tool result. The
  direct `available_functions[...]` call path is kept as a fallback for any
  caller that still passes it directly (e.g. a non-native/text-tool-calling
  path), so this still resolves in one call() when that happens.
"""
import json
from typing import Any, Mapping, Optional, Sequence, Union

from crewai.llms.base_llm import BaseLLM

from llm.mock_llm import MockBrain, ToolSpec


def _parse_tool_specs(tools: Optional[list[dict]]) -> list[ToolSpec]:
    specs = []
    for t in tools or []:
        fn = t.get("function", t)
        name = fn.get("name")
        params = fn.get("parameters", {}) or {}
        arg_fields = frozenset((params.get("properties") or {}).keys())
        if name:
            specs.append(ToolSpec(name=name, arg_fields=arg_fields))
    return specs


def _first_user_content(messages: Sequence[Mapping[str, Any]]) -> str:
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "user":
            content = m.get("content")
            if isinstance(content, str):
                return content
    return ""


class MockLLM(BaseLLM):
    def __init__(self, agent_role: str, model: str = "mock-model"):
        super().__init__(model=model, temperature=0.0)
        self.agent_role = agent_role
        self.brain = MockBrain()

    def call(
        self,
        messages,
        tools: Optional[list[dict]] = None,
        callbacks: Optional[list] = None,
        available_functions: Optional[dict] = None,
        from_task: Optional[Any] = None,
        from_agent: Optional[Any] = None,
        response_model: Optional[type] = None,
    ) -> Union[str, list[dict]]:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        last_message = messages[-1] if messages else {}
        if not isinstance(last_message, dict):
            last_message = {}

        if last_message.get("role") == "tool":
            # CrewAI already executed the tool call we returned below and
            # appended the real result as the latest message -- compose the
            # final answer from that real result, not from a re-decision.
            task_text = _first_user_content(messages)
            tool_result = last_message.get("content")
            tool_result = tool_result if isinstance(tool_result, str) else ""
            tool_name = last_message.get("name") or ""
            return self.brain.compose_final_answer(self.agent_role, task_text, tool_name, tool_result)

        content = last_message.get("content")
        task_text = content if isinstance(content, str) else ""

        tool_specs = _parse_tool_specs(tools)
        decision = self.brain.decide(self.agent_role, task_text, tool_specs)

        invocation = decision.tool_invocation
        if decision.kind == "tool_call" and invocation is not None:
            if available_functions:
                fn = available_functions.get(invocation.tool_name)
                if fn is not None:
                    tool_result = fn(**invocation.arguments)
                    return self.brain.compose_final_answer(
                        self.agent_role, task_text, invocation.tool_name, tool_result
                    )

            # Native tool-calling protocol (see module docstring): hand back a
            # tool-call request and let CrewAI execute the real tool itself.
            return [
                {
                    "id": f"call_{invocation.tool_name}",
                    "type": "function",
                    "function": {
                        "name": invocation.tool_name,
                        "arguments": json.dumps(invocation.arguments),
                    },
                }
            ]

        return decision.final_text or self.brain.compose_final_answer(self.agent_role, task_text, None, None)

    def supports_function_calling(self) -> bool:
        return True

    def supports_stop_words(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return 8192
