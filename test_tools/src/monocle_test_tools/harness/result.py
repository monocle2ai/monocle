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


class ScenarioResult(BaseModel):
    """Turns and spans stay out of ``repr``, so a failed assert prints the failure, not every span."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    test_name: str
    scenario: str
    passed: bool
    turns_used: int
    max_turns: int
    failure_reason: Optional[str] = None
    turns: list[TurnRecord] = Field(default_factory=list, repr=False)
    missing_required_params: list[str] = Field(default_factory=list)
    spans: tuple = Field((), repr=False)                    # the agent's spans across every turn
    per_turn_spans: list[tuple] = Field(default_factory=list, repr=False)

    def report(self) -> str:
        """The run as a transcript, used as the failure message."""
        lines = [f"{'PASSED' if self.passed else 'FAILED'} scenario '{self.scenario}' "
                 f"after {self.turns_used}/{self.max_turns} turns"]
        if self.failure_reason:
            lines.append(f"  reason: {self.failure_reason}")
        for record in self.turns:
            prefix = f"  turn {record.turn}  "
            indent = " " * len(prefix)
            for call in record.param_tools_called:
                lines.append(f"{prefix}tester> [called {call.name}()] ==> {call.value}")
                prefix = indent
            lines.append(f"{prefix}tester> {record.tester_message}")
            lines.append(f"{indent}target> {record.target_response}")
            if record.verdict is not None:
                lines.append(f"{indent}verdict> {'met' if record.verdict.met else 'not met'} -- {record.verdict.reason}")
        return "\n".join(lines)
