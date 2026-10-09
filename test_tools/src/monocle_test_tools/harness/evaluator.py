"""LLM judge: does the target agent's response satisfy the success criteria?

Used at two call sites. The driver calls :meth:`ResponseEvaluator.judge` after every
target response -- that verdict is authoritative and decides the test. The same
evaluator is also bound onto the test agent as an ``evaluate_response`` tool so it can
self-check mid-turn and adapt its next message; that verdict decides nothing.
"""
from typing import Any, Iterable, Optional

from pydantic import BaseModel, Field

from monocle_test_tools.trace_utils import get_tool_invocations

JUDGE_PREAMBLE = """You judge whether a target agent's response satisfies a test's \
success criteria.

Be strict. The criteria are met only when the response shows they are met. An agent \
promising to do something, asking a follow-up question, or describing what it is about \
to do does NOT meet criteria that require the thing to be done. When the criteria instead \
describe a behaviour (asking before acting, confirming, refusing), judge whether the \
response shows that behaviour. Judge only what the response actually establishes.

When the tool calls the agent made this turn are listed below the response, a claim that \
something was done (booked, sent, saved) counts only if a matching tool call is listed."""


def render_tool_calls(spans: Iterable[Any]) -> str:
    """One line per tool the agent invoked, with what went in and what came out."""
    return "\n".join(f"- {name}(input={input_text}) -> {output_text}"
                     for name, input_text, output_text in get_tool_invocations(spans))


class Verdict(BaseModel):
    """The judge's decision about one target-agent response."""

    met: bool = Field(..., description="True when the success criteria are satisfied.")
    reason: str = Field(..., description="One sentence explaining the verdict.")
    confidence: Optional[float] = Field(
        None, description="Confidence between 0.0 and 1.0.")


class ResponseEvaluator:
    """Judges target-agent responses against a scenario's success criteria."""

    def __init__(self, model: Any, success_criteria: str):
        self._model = model
        self._success_criteria = success_criteria

    def build_judge_prompt(self, target_response: Any, transcript: str = "",
                           tool_calls: Optional[str] = None) -> str:
        """Assemble the judge prompt. Pure -- no model call.

        ``tool_calls`` is the rendered list for this turn; ``None`` means the caller
        has no span information, so the section is left out rather than shown empty.
        """
        lines = [
            JUDGE_PREAMBLE,
            "",
            "[SUCCESS CRITERIA]",
            self._success_criteria,
            "",
            "[CONVERSATION SO FAR]",
            transcript or "(none)",
            "",
            "[LATEST TARGET AGENT RESPONSE]",
            str(target_response),
        ]
        if tool_calls is not None:
            lines += ["", "[TOOL CALLS THIS TURN]", tool_calls or "(none)"]
        return "\n".join(lines)

    async def judge(self, target_response: Any, transcript: str = "",
                    spans: Optional[Iterable[Any]] = None) -> Verdict:
        """Ask the judge model whether the criteria are met.

        ``spans`` are the target's spans for this turn; the tool calls in them are
        shown to the judge so "booked" is checked against a real booking call. The
        tester's self-check tool has no spans and passes ``None``.
        """
        tool_calls = render_tool_calls(spans) if spans is not None else None
        structured = self._model.with_structured_output(Verdict)
        return await structured.ainvoke(
            self.build_judge_prompt(target_response, transcript, tool_calls))

    def as_tool(self) -> Any:
        """Bindable ``evaluate_response`` tool wrapping the same judge."""
        from langchain_core.tools import StructuredTool

        async def evaluate_response(target_response: str) -> str:
            verdict = await self.judge(target_response)
            state = "met" if verdict.met else "not met"
            return f"Success criteria {state}: {verdict.reason}"

        return StructuredTool.from_function(
            coroutine=evaluate_response,
            name="evaluate_response",
            description=(
                "Check whether the target agent's latest response satisfies the test's "
                "success criteria. Pass the target agent's response. Returns the "
                "verdict and the reason for it."
            ),
        )
