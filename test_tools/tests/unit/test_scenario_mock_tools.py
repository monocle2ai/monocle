"""A scenario can carry mock tools: the harness applies them on every turn of its conversation."""
import asyncio
import json

from langchain_core.messages import AIMessage
from opentelemetry.context import attach, detach
from opentelemetry.sdk.trace import TracerProvider

from monocle_apptrace.instrumentation.common.span_handler import SpanHandler
from monocle_test_tools import MockTool, MonocleValidator
from monocle_test_tools.harness import driver
from monocle_test_tools.harness.evaluator import Verdict
from monocle_test_tools.harness.scenario import ScenarioTestCase
from monocle_test_tools.scenarios.scenario import scenarios_from

_tracer = TracerProvider().get_tracer("scenario-mock-tool-tests")

SCENARIO = {
    "scenario": "Check the status of my order.",
    "persona": "A customer who repeats the question",
    "success_criteria": "The assistant reports the order status.",
    "max_turns": 2,
    "target_description": "An order status assistant.",
    "mock_tools": [{"name": "get_status", "type": "tool.adk", "response": "shipped"}],
}


def test_a_scenario_carries_its_mock_tools():
    case = ScenarioTestCase.model_validate(SCENARIO)
    assert isinstance(case.mock_tools[0], MockTool)
    assert case.mock_tools[0].response == "shipped"


def test_a_scenario_without_mock_tools_has_none():
    case = ScenarioTestCase.model_validate({k: v for k, v in SCENARIO.items() if k != "mock_tools"})
    assert case.mock_tools == []


def test_mock_tools_load_from_a_scenario_file(tmp_path):
    (tmp_path / "cases.json").write_text(json.dumps({"scenarios": [{**SCENARIO, "test_name": "order_status"}]}))
    [scenario] = scenarios_from("cases.json", str(tmp_path))
    assert scenario.mock_tools[0].name == "get_status"


class _Tester:
    """The simulated user: says the same thing every turn."""
    async def ainvoke(self, state):
        return {"messages": list(state["messages"]) + [AIMessage(content="Where is my order?")]}


class _Judge:
    """Never satisfied, so the conversation runs all of its turns."""
    def __init__(self, model, success_criteria):
        pass

    async def judge(self, response, transcript, spans=()):
        return Verdict(met=False, reason="keep going")


class _Runner:
    async def end_session(self, session_id):
        return None


def _run_scenario(monkeypatch, case):
    """Run the real harness loop; each agent turn makes one get_status call through the mocks it was given."""
    validator = MonocleValidator()
    responses = []

    async def agent_turn(agent, agent_type, *args, mock_tools=(), **kwargs):
        context = validator._set_wrapper_methods(list(mock_tools))
        token = attach(context) if context is not None else None
        try:
            with _tracer.start_as_current_span("get_status") as span:
                span.set_attribute("span.type", "agentic.tool.invocation")
                span.set_attribute("entity.1.name", "get_status")
                span.set_attribute("entity.1.type", "tool.adk")
                skipped, response = SpanHandler.skip_execution(span)
        finally:
            if token is not None:
                detach(token)
        responses.append(response if skipped else "real tool ran")
        return "Your order status is " + str(responses[-1])

    monkeypatch.setattr(driver, "resolve_model", lambda spec, *a, **k: object())
    monkeypatch.setattr(driver, "build_test_agent", lambda case, description, model: (_Tester(), []))
    monkeypatch.setattr(driver, "get_agent_runner", lambda agent_type: _Runner())
    monkeypatch.setattr(driver, "ResponseEvaluator", _Judge)
    monkeypatch.setattr(validator, "run_agent_async", agent_turn)
    harness = driver.ScenarioHarness(case, model=object())
    asyncio.run(harness.run_scenario_async(object(), "google_adk", "An order status assistant."))
    return responses


def test_the_harness_applies_the_mocks_on_every_turn(monkeypatch):
    case = ScenarioTestCase.model_validate(SCENARIO)
    assert _run_scenario(monkeypatch, case) == ["shipped", "shipped"]


def test_without_mock_tools_the_real_tool_runs(monkeypatch):
    case = ScenarioTestCase.model_validate({k: v for k, v in SCENARIO.items() if k != "mock_tools"})
    assert _run_scenario(monkeypatch, case) == ["real tool ran", "real tool ran"]
