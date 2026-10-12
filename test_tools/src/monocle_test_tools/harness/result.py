"""What a scenario run produces: the per-turn record and the overall result."""
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from monocle_test_tools.harness.evaluator import Verdict
from monocle_test_tools.harness.param_tools import ParamCall


class TurnRecord(BaseModel):
    """One exchange: what the test agent said, what the target replied, the verdict."""

    turn: int = Field(..., description="1-based turn number.")
    tester_message: str = Field(..., description="Message sent to the target agent.")
    target_response: Any = Field(None, description="The target agent's reply.")
    verdict: Optional[Verdict] = Field(
        None, description="Driver's verdict; None when the turn ended in an error.")
    param_tools_called: list[ParamCall] = Field(
        default_factory=list,
        description="Param tools the test agent called this turn, with their values.")


class ScenarioResult(BaseModel):
    """The outcome of one scenario run.

    ``turns``, ``spans`` and ``per_turn_spans`` are excluded from ``repr`` on purpose.
    A failing ``assert result.passed`` makes pytest print ``repr(result)``, and the
    default Pydantic repr dumps every span object and every message -- hundreds of lines
    of ``<ReadableSpan object at 0x...>`` burying the actual failure. The transcript
    belongs in :meth:`report`, which is what callers pass as the assert message; the
    fields are still fully readable as attributes.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    test_name: str
    scenario: str
    passed: bool
    turns_used: int
    max_turns: int
    failure_reason: Optional[str] = Field(
        None, description='"max turns exhausted" or "target error: ...".')
    turns: list[TurnRecord] = Field(default_factory=list, repr=False)
    missing_required_params: list[str] = Field(
        default_factory=list,
        description="Required params the target agent never asked for. Non-empty fails "
                    "the scenario even when the judge is satisfied.")
    spans: tuple = Field((), repr=False,
                         description="Target spans across every turn.")
    per_turn_spans: list[tuple] = Field(default_factory=list, repr=False,
                                        description="Target spans, one tuple per turn.")

    def report(self) -> str:
        """Render the run as a transcript, for use as an assertion message."""
        outcome = "PASSED" if self.passed else "FAILED"
        lines = [f"{outcome} scenario '{self.scenario}' after "
                 f"{self.turns_used}/{self.max_turns} turns"]
        if self.failure_reason:
            lines.append(f"  reason: {self.failure_reason}")
        if self.missing_required_params:
            lines.append("  never requested: "
                         + ", ".join(self.missing_required_params))
        if self.turns and self.turns[-1].verdict is not None:
            lines.append(f"  last verdict: {self.turns[-1].verdict.reason}")
        for record in self.turns:
            # The turn number labels the turn's first line, and every following line of
            # that turn is indented to line up under it. Each tool call gets its own
            # line showing what it returned, so the tester's message stays verbatim --
            # what actually went to the target agent, with nothing prepended to it.
            prefix = f"  turn {record.turn}  "
            indent = " " * len(prefix)
            for call in record.param_tools_called:
                lines.append(f"{prefix}tester> [called {call.name}()] ==> {call.value}")
                prefix = indent
            lines.append(f"{prefix}tester> {record.tester_message}")
            lines.append(f"{indent}target> {record.target_response}")
            if record.verdict is not None:
                state = "met" if record.verdict.met else "not met"
                lines.append(f"{indent}verdict> {state} -- {record.verdict.reason}")
        return "\n".join(lines)
