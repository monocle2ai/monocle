"""LLM judge: does the target agent's response satisfy the success criteria?

Used at two call sites. The driver calls :meth:`ResponseEvaluator.judge` after every
target response -- that verdict is authoritative and decides the test. The same
evaluator is also bound onto the test agent as an ``evaluate_response`` tool so it can
self-check mid-turn and adapt its next message; that verdict decides nothing.
"""
from typing import Any, Optional

from pydantic import BaseModel, Field

JUDGE_PREAMBLE = """You judge whether a target agent's response satisfies a test's \
success criteria.

Be strict. The criteria are met only when the response shows they are met. An agent \
promising to do something, asking a follow-up question, or describing what it is about \
to do does NOT meet criteria that require the thing to be done. Judge only what the \
response actually establishes."""


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

    def build_judge_prompt(self, target_response: Any, transcript: str = "") -> str:
        """Assemble the judge prompt. Pure -- no model call."""
        return "\n".join([
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
        ])

    async def judge(self, target_response: Any, transcript: str = "") -> Verdict:
        """Ask the judge model whether the criteria are met."""
        structured = self._model.with_structured_output(Verdict)
        return await structured.ainvoke(
            self.build_judge_prompt(target_response, transcript))

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
