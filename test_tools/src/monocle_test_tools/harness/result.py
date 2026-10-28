"""What a scenario run produces."""
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from monocle_test_tools.harness.evaluator import Verdict
from monocle_test_tools.harness.test_agent import ParamCall


class TurnRecord(BaseModel):
    turn: int
    tester_message: str
    target_response: Any = None
    verdict: Optional[Verdict] = None                       # None when the agent errored
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
    success_criteria: Optional[str] = None
    session_id: Optional[str] = None
    tester_model: Optional[str] = None
    judge_model: Optional[str] = None
    turns: list[TurnRecord] = Field(default_factory=list, repr=False)
    missing_required_params: list[str] = Field(default_factory=list)
    spans: tuple = Field((), repr=False)                    # the agent's spans across every turn
    per_turn_spans: list[tuple] = Field(default_factory=list, repr=False)

    @property
    def headline(self) -> str:
        return "passed" if self.passed else (self.failure_reason or "failed")

    def report(self, saved: Optional[str] = None) -> str:
        """The failure message: why it failed, where to look, then the conversation."""
        details = [("test", self.test_name), ("turns", f"{self.turns_used} of {self.max_turns}"),
                   ("criteria", self.success_criteria),
                   ("models", self.tester_model and f"tester {self.tester_model}, judge {self.judge_model}"),
                   ("session", self.session_id), ("saved", saved)]
        lines = [self.headline, ""]
        lines += [_field(f"  {label:<10}", value) for label, value in details if value]
        for record in self.turns:
            lines += ["", f"  turn {record.turn}" + (f"  trace {record.trace_id}" if record.trace_id else "")]
            for call in record.param_tools_called:
                lines.append(_field("    param  ", f"{call.name} = {call.value}"))
            lines.append(_field("    user   ", record.tester_message))
            if record.verdict is None:
                lines.append(_field("    agent  ", "(raised an error)"))
                continue
            lines.append(_field("    agent  ", record.target_response))
            outcome = "violated" if record.verdict.violated else "met" if record.verdict.met else "not met"
            lines.append(_field("    judge  ", f"[{outcome}] {record.verdict.reason}"))
        return "\n".join(lines)


def _field(label: str, value: Any) -> str:
    """``label`` then ``value``, with any further lines of ``value`` lined up under the first."""
    return label + str(value).replace("\n", "\n" + " " * len(label))
