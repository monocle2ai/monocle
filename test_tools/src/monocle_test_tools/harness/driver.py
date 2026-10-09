"""The scenario driver: coordinates the test agent and the target agent.

The driver owns the loop. Each turn it asks the test agent for the next user message,
sends that message to the target agent through ``MonocleValidator.run_agent_async``, and
judges the reply against the scenario's success criteria. Owning the loop here -- rather
than letting the test agent call the target from a tool -- is what keeps turn counting,
abort semantics, per-turn span snapshots and session cleanup in one place.
"""
import logging
import os
from contextlib import contextmanager
from typing import Any, Optional, Union

from opentelemetry.context import attach, detach, set_value

from monocle_apptrace.instrumentation.common.constants import MONOCLE_SUPPRESS_SPANS
from monocle_test_tools.harness.evaluator import ResponseEvaluator
from monocle_test_tools.harness.result import ScenarioResult, TurnRecord
from monocle_test_tools.harness.scenario import ScenarioTestCase
from monocle_test_tools.harness.test_agent import build_test_agent
from monocle_test_tools.runner.runner import get_agent_runner
from monocle_test_tools.validator import MonocleValidator

logger = logging.getLogger(__name__)

TEST_AGENT_MODEL_ENV = "MONOCLE_TEST_AGENT_MODEL"
TEST_JUDGE_MODEL_ENV = "MONOCLE_TEST_JUDGE_MODEL"
TRACE_TEST_AGENT_ENV = "MONOCLE_TRACE_TEST_AGENT"

# Some providers reject an empty message list, so turn 1 starts from this.
KICKOFF = ("Start the conversation with the target agent. Send your first message now, "
           "in persona.")


class ScenarioHarness:
    """Runs one scenario against one target agent."""

    def __init__(self, case: Union[ScenarioTestCase, dict], model: Any = None,
                 judge_model: Any = None, trace_test_agent: Optional[bool] = None):
        self.case = (ScenarioTestCase.model_validate(case)
                     if isinstance(case, dict) else case)
        self._model_spec = model
        self._judge_model_spec = judge_model
        if trace_test_agent is None:
            trace_test_agent = os.getenv(TRACE_TEST_AGENT_ENV, "false").lower() == "true"
        self.trace_test_agent = trace_test_agent
        self.validator = MonocleValidator()   # the singleton the fixture set up

    @staticmethod
    def _resolve_model(spec: Any, env_var: str, fallback: Optional[str] = None) -> Any:
        """Resolve a model spec: an explicit model object, a model string, or an env var."""
        from langchain.chat_models import init_chat_model

        if spec is None:
            spec = os.getenv(env_var) or fallback
        if spec is None:
            raise ValueError(
                f"No model configured for the scenario harness. Set {env_var} "
                f"(e.g. 'openai:gpt-4.1') or pass one to ScenarioHarness().")
        if isinstance(spec, str):
            return init_chat_model(spec)
        return spec

    @contextmanager
    def _suppress_spans(self):
        """Emit no spans for the enclosed calls -- the harness's own LLM traffic."""
        if self.trace_test_agent:
            yield
            return
        token = attach(set_value(MONOCLE_SUPPRESS_SPANS, True))
        try:
            yield
        finally:
            detach(token)

    @staticmethod
    def _transcript(records: list[TurnRecord], current_turn: int, current_message: str) -> str:
        """The conversation so far, ending with what the user just said this turn."""
        lines = []
        for record in records:
            lines.append(f"turn {record.turn} user> {record.tester_message}")
            lines.append(f"turn {record.turn} agent> {record.target_response}")
        lines.append(f"turn {current_turn} user> {current_message}")
        return "\n".join(lines)

    async def run_scenario_async(self, target_agent: Any, agent_type: str,
                                 target_description: str) -> ScenarioResult:
        """Run the scenario. Never raises on a behavioral failure."""
        from langchain_core.messages import HumanMessage

        case = self.case
        model = self._resolve_model(self._model_spec, TEST_AGENT_MODEL_ENV)
        judge = self._resolve_model(self._judge_model_spec, TEST_JUDGE_MODEL_ENV,
                                    fallback=os.getenv(TEST_AGENT_MODEL_ENV))
        evaluator = ResponseEvaluator(judge, case.success_criteria)
        tester, param_calls = build_test_agent(case, target_description, model)

        messages: list[Any] = [HumanMessage(content=KICKOFF)]
        records: list[TurnRecord] = []
        per_turn_spans: list[tuple] = []
        passed = False
        failure_reason: Optional[str] = None

        try:
            for turn in range(1, case.max_turns + 1):
                param_calls.clear()
                with self._suppress_spans():
                    state = await tester.ainvoke({"messages": messages})
                tester_message = state["messages"][-1].content
                called = list(param_calls)

                try:
                    target_response = await self.validator.run_agent_async(
                        target_agent, agent_type, tester_message,
                        session_id=case.session_id)
                except Exception as error:  # pylint: disable=broad-except
                    failure_reason = f"target error: {error}"
                    records.append(TurnRecord(turn=turn, tester_message=tester_message,
                                              param_tools_called=called))
                    break

                turn_spans = self.validator.spans
                per_turn_spans.append(turn_spans)

                with self._suppress_spans():
                    verdict = await evaluator.judge(target_response,
                                                    self._transcript(records, turn, tester_message),
                                                    spans=turn_spans)

                records.append(TurnRecord(
                    turn=turn, tester_message=tester_message,
                    target_response=target_response, verdict=verdict,
                    param_tools_called=called))

                if verdict.violated:
                    failure_reason = f"criteria violated: {verdict.reason}"
                    break
                if verdict.met and turn >= case.min_turns:
                    passed = True
                    break

                # Keep the tester's tool calls in its history, or it starts inventing
                # values instead of calling the tools (10/10 vs 4/10 in testing).
                # Verdicts stay out: the Human role in this history is the target.
                messages = list(state["messages"])
                messages.append(HumanMessage(content=str(target_response)))
            else:
                failure_reason = "max turns exhausted"
        finally:
            try:
                await get_agent_runner(agent_type).end_session(case.session_id)
            except Exception as error:  # pylint: disable=broad-except
                logger.debug("end_session cleanup failed: %s", error)

        # Met the criteria without asking for a required detail = invented it.
        requested = {call.name for record in records
                     for call in record.param_tools_called}
        missing = case.missing_required_params(requested)
        if missing:
            passed = False
            clause = f"required params never requested: {', '.join(missing)}"
            failure_reason = f"{failure_reason}; {clause}" if failure_reason else clause

        # Let the fixture assert over the whole run, as test_multi_turn_agent_async does.
        all_spans = tuple(self.validator._test_all_up_spans)  # pylint: disable=protected-access
        self.validator._spans = all_spans  # pylint: disable=protected-access

        return ScenarioResult(
            test_name=case.test_name, scenario=case.scenario, passed=passed,
            turns_used=len(records), max_turns=case.max_turns,
            failure_reason=failure_reason, turns=records,
            missing_required_params=missing,
            spans=all_spans, per_turn_spans=per_turn_spans)

    async def test_scenario_async(self, target_agent: Any, agent_type: str,
                                  target_description: str) -> ScenarioResult:
        """Run the scenario and assert it passed, reporting the transcript on failure."""
        result = await self.run_scenario_async(target_agent, agent_type,
                                               target_description)
        assert result.passed, result.report()
        return result
