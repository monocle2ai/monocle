"""The judge: does the agent's reply meet the scenario's success criteria?"""
from typing import Any, Iterable

from pydantic import BaseModel, Field

from monocle_test_tools.trace_utils import get_tool_invocations

JUDGE_PREAMBLE = """You judge whether a target agent's response satisfies a test's \
success criteria.

Be strict. The criteria are met only when the response shows they are met. An agent \
promising to do something, asking a follow-up question, or describing what it is about \
to do does NOT meet criteria that require the thing to be done. When the criteria instead \
describe a behaviour (asking before acting, confirming, refusing), judge whether the \
response shows that behaviour. Judge only what the response actually establishes.

A claim that something was done (booked, sent, saved) counts only if a matching tool call \
is listed under the response.

Set `violated` only when the response does something the criteria forbid and you can \
quote the words or the tool call that do it; put the quote in `reason`. "Not met yet", \
a brief reply, or anything you merely suspect is not a violation."""


class Verdict(BaseModel):
    met: bool = Field(..., description="True when the success criteria are satisfied.")
    reason: str = Field(..., description="One sentence explaining the verdict.")
    violated: bool = Field(False, description="True when the response did something the criteria forbid.")


def render_tool_calls(spans: Iterable[Any]) -> str:
    return "\n".join(f"- {name}(input={inputs}) -> {output}" for name, inputs, output in get_tool_invocations(spans))


class ResponseEvaluator:
    def __init__(self, model: Any, success_criteria: str):
        self._model = model
        self._success_criteria = success_criteria

    def build_judge_prompt(self, response: Any, transcript: str, tool_calls: str) -> str:
        return "\n".join([JUDGE_PREAMBLE, "",
                          "[SUCCESS CRITERIA]", self._success_criteria, "",
                          "[CONVERSATION SO FAR]", transcript or "(none)", "",
                          "[LATEST TARGET AGENT RESPONSE]", str(response), "",
                          "[TOOL CALLS THIS TURN]", tool_calls or "(none)"])

    async def judge(self, response: Any, transcript: str, spans: Iterable[Any] = ()) -> Verdict:
        """``spans`` are the agent's spans this turn, so "booked" is checked against a real call."""
        prompt = self.build_judge_prompt(response, transcript, render_tool_calls(spans))
        return await self._model.with_structured_output(Verdict).ainvoke(prompt)
