"""Runs one scenario: the simulated user talks to the agent, and a judge checks every reply."""
import logging
import os
from contextlib import contextmanager
from typing import Any, Optional, Union

from opentelemetry.context import attach, detach, set_value

try:
    from monocle_apptrace.instrumentation.common.constants import MONOCLE_SUPPRESS_SPANS
except ImportError:  # an older monocle_apptrace; only scenario runs need it, so fail there
    MONOCLE_SUPPRESS_SPANS = None
from monocle_test_tools.harness.evaluator import ResponseEvaluator
from monocle_test_tools.harness.result import ScenarioResult, TurnRecord
from monocle_test_tools.harness.scenario import ScenarioTestCase
from monocle_test_tools.harness.test_agent import build_test_agent, from_target
from monocle_test_tools.runner.runner import get_agent_runner
from monocle_test_tools.validator import MonocleValidator

logger = logging.getLogger(__name__)

TEST_AGENT_MODEL_ENV = "MONOCLE_TEST_AGENT_MODEL"
TEST_JUDGE_MODEL_ENV = "MONOCLE_TEST_JUDGE_MODEL"
TRACE_TEST_AGENT_ENV = "MONOCLE_TRACE_TEST_AGENT"

# Some providers reject an empty message list, so turn 1 starts from this.
KICKOFF = "Start the conversation with the target agent. Send your first message now, in persona."


def resolve_model(spec: Any, env_var: str = TEST_AGENT_MODEL_ENV, fallback: Any = None) -> Any:
    """A model object, a ``provider:model`` string, ``local`` / ``local:<name>``, the env var, or ``fallback``."""
    spec = spec if spec is not None else (os.getenv(env_var) or fallback)
    if spec is None:
        raise ValueError(f"No model configured for scenario tests. Pass model= (e.g. 'openai:gpt-4.1-mini'), "
                         f"set {env_var}, or use 'local' to run an open model on this machine with no API key.")
    if not isinstance(spec, str):
        return spec
    from monocle_test_tools.harness.local_model import is_local_spec, local_chat_model
    if is_local_spec(spec):
        return local_chat_model(spec)
    from langchain.chat_models import init_chat_model
    return init_chat_model(spec)


def trace_id_of(spans: tuple) -> Optional[str]:
    return format(spans[0].get_span_context().trace_id, "032x") if spans else None


class ScenarioHarness:
    def __init__(self, case: Union[ScenarioTestCase, dict], model: Any = None,
                 judge_model: Any = None, trace_test_agent: Optional[bool] = None):
        self.case = ScenarioTestCase.model_validate(case) if isinstance(case, dict) else case
        self._model_spec = model
        self._judge_model_spec = judge_model
        if trace_test_agent is None:
            trace_test_agent = os.getenv(TRACE_TEST_AGENT_ENV, "false").lower() == "true"
        self.trace_test_agent = trace_test_agent
        self.validator = MonocleValidator()

    @contextmanager
    def _suppress_spans(self):
        """Keep the simulated user's and the judge's own model calls out of the traces."""
        if self.trace_test_agent:
            yield
            return
        if MONOCLE_SUPPRESS_SPANS is None:
            raise RuntimeError("Scenario tests need a newer monocle_apptrace, one that can suppress the "
                               "test agent's spans. Run: pip install -U monocle_apptrace")
        token = attach(set_value(MONOCLE_SUPPRESS_SPANS, True))
        try:
            yield
        finally:
            detach(token)

    @staticmethod
    def _transcript(records: list[TurnRecord], turn: int, message: str) -> str:
        """The conversation so far, ending with what the user just said."""
        lines = []
        for record in records:
            lines += [f"turn {record.turn} user> {record.tester_message}",
                      f"turn {record.turn} agent> {record.target_response}"]
        lines.append(f"turn {turn} user> {message}")
        return "\n".join(lines)

    async def run_scenario_async(self, target_agent: Any, agent_type: str,
                                 target_description: str) -> ScenarioResult:
        """Run the conversation. Returns the result; never raises on a failed scenario."""
        from langchain_core.messages import HumanMessage

        case = self.case
        model = resolve_model(self._model_spec, TEST_AGENT_MODEL_ENV)
        evaluator = None
        if case.success_criteria is not None:
            judge = resolve_model(self._judge_model_spec, TEST_JUDGE_MODEL_ENV, fallback=model)
            evaluator = ResponseEvaluator(judge, case.success_criteria)
        tester, param_calls = build_test_agent(case, target_description, model)
        runner = get_agent_runner(agent_type)   # one for the whole conversation: ADK keeps session state on it

        messages: list[Any] = [HumanMessage(content=KICKOFF)]
        records: list[TurnRecord] = []
        per_turn_spans: list[tuple] = []
        passed, failure_reason, reason = False, None, None
        try:
            for turn in range(1, case.max_turns + 1):
                param_calls.clear()
                with self._suppress_spans():
                    state = await tester.ainvoke({"messages": messages})
                tester_message = state["messages"][-1].content
                called = list(param_calls)

                try:
                    reply = await self.validator.run_agent_async(
                        target_agent, agent_type, tester_message, session_id=case.session_id, agent_runner=runner)
                except Exception as error:  # pylint: disable=broad-except
                    failure_reason, reason = f"agent raised an error at turn {turn} of {case.max_turns}", str(error)
                    records.append(TurnRecord(turn=turn, tester_message=tester_message, param_tools_called=called))
                    break

                spans = self.validator.spans
                per_turn_spans.append(spans)
                verdict = None
                if evaluator:
                    with self._suppress_spans():
                        verdict = await evaluator.judge(reply, self._transcript(records, turn, tester_message), spans=spans)
                records.append(TurnRecord(turn=turn, tester_message=tester_message, target_response=reply,
                                          verdict=verdict, param_tools_called=called, trace_id=trace_id_of(spans)))

                if verdict and verdict.violated:
                    failure_reason = f"criteria violated at turn {turn} of {case.max_turns}"
                    reason = verdict.reason
                    break
                if verdict and verdict.met and turn >= case.min_turns:
                    passed = True
                    break
                # The tester's tool calls stay in its history, or it starts inventing values
                # instead of calling the tools. The target's reply comes back as the next user turn,
                # marked as the target's so the tester doesn't answer it in the target's role.
                messages = list(state["messages"]) + [HumanMessage(content=from_target(reply))]
            else:
                if evaluator:
                    failure_reason = f"criteria not met in {case.max_turns} turns"
                    reason = records[-1].verdict.reason if records else None
                else:
                    passed = True       # nothing to judge: a conversation that ran all its turns passes
        finally:
            try:
                await runner.end_session(case.session_id)
            except Exception as error:  # pylint: disable=broad-except
                logger.warning("end_session cleanup failed for %s: %s", case.session_id, error)

        # Done without asking for a held-back detail means the agent made it up.
        requested = {call.name for record in records for call in record.param_tools_called}
        missing = case.missing_required_params(requested)
        if missing:
            passed = False
            clause = f"never asked for {', '.join(missing)}"
            failure_reason = f"{failure_reason}; {clause}" if failure_reason else clause

        # Let the trace-asserter fixture check the whole conversation, not just the last turn.
        all_spans = tuple(self.validator._test_all_up_spans)  # pylint: disable=protected-access
        self.validator._spans = all_spans  # pylint: disable=protected-access

        return ScenarioResult(test_name=case.test_name, scenario=case.scenario, passed=passed,
                              turns_used=len(records), max_turns=case.max_turns, failure_reason=failure_reason,
                              reason=reason, success_criteria=case.success_criteria, session_id=case.session_id,
                              turns=records, missing_required_params=missing,
                              spans=all_spans, per_turn_spans=per_turn_spans)
