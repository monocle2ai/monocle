"""What a scenario run produces."""
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from monocle_test_tools.harness.evaluator import Verdict
from monocle_test_tools.harness.test_agent import ParamCall


class TurnRecord(BaseModel):
    turn: int
    tester_message: str
    target_response: Any = None
    verdict: Optional[Verdict] = None                       # None when the agent errored or nothing is judged
    param_tools_called: list[ParamCall] = Field(default_factory=list)
    trace_id: Optional[str] = None                          # the agent's trace for this turn


class ScenarioResult(BaseModel):
    """Turns and spans stay out of ``repr``, so a failed assert prints the failure, not every span."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    test_name: str
    scenario: str
    passed: bool
    turns_used: int
    max_turns: int
    failure_reason: Optional[str] = None
    reason: Optional[str] = None                            # the judge's explanation, or the agent's error
    success_criteria: Optional[str] = None
    session_id: Optional[str] = None
    turns: list[TurnRecord] = Field(default_factory=list, repr=False)
    missing_required_params: list[str] = Field(default_factory=list)
    spans: tuple = Field((), repr=False)                    # the agent's spans across every turn
    per_turn_spans: list[tuple] = Field(default_factory=list, repr=False)

    def report(self, saved: Optional[str] = None) -> str:
        """The failure message: what went wrong, where to look, then the conversation."""
        lines = [self.failure_reason or ("passed" if self.passed else "failed")]
        lines += [_field(f"  {label:<10}", value) for label, value in
                  [("reason", self.reason), ("criteria", self.success_criteria),
                   ("session", self.session_id), ("saved", saved)] if value]
        for record in self.turns:
            lines += ["", f"  turn {record.turn}" + (f"  trace {record.trace_id}" if record.trace_id else "")]
            for call in record.param_tools_called:
                lines.append(_field("    param  ", f"{call.name} = {call.value}"))
            lines.append(_field("    user   ", record.tester_message))
            if record.verdict is None and record.target_response is None:
                lines.append("    agent  (error, see reason)")
                continue
            lines.append(_field("    agent  ", record.target_response))
            if record.verdict is None:      # an unjudged conversation
                continue
            outcome = "violated" if record.verdict.violated else "met" if record.verdict.met else "not met"
            lines.append(_field("    judge  ", f"[{outcome}] {record.verdict.reason}"))
        return "\n".join(lines)


def _field(label: str, value: Any) -> str:
    """``label`` then ``value``, with any further lines of ``value`` lined up under the first."""
    return label + str(value).replace("\n", "\n" + " " * len(label))
