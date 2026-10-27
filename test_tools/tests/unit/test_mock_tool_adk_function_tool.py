"""End to end through Monocle's real instrumentation: a mocked ADK FunctionTool answers each call
from the mock's responses or handler, and the real function never runs. No model is involved."""
import asyncio

import pytest
from opentelemetry.context import attach, detach

pytest.importorskip("google.adk")
from google.adk.tools import FunctionTool  # noqa: E402

from monocle_test_tools import MockTool, MonocleValidator  # noqa: E402

real_calls = []


def get_order_status(order_id: str) -> dict:
    """The real tool; a mocked run must never reach it."""
    real_calls.append(order_id)
    return {"status": "REAL"}


def _call_tool_in_one_run(validator, mock_tools, order_ids):
    tool = FunctionTool(get_order_status)
    token = attach(validator._set_wrapper_methods(mock_tools))
    try:
        return [asyncio.run(tool.run_async(args={"order_id": order_id}, tool_context=None)) for order_id in order_ids]
    finally:
        detach(token)


@pytest.fixture
def validator():
    validator = MonocleValidator()
    validator.reset_mock_tool_state()
    real_calls.clear()
    yield validator
    validator.reset_mock_tool_state()


def test_a_mocked_function_tool_returns_its_responses_in_call_order(validator):
    mock = MockTool(name="get_order_status", type="tool.adk",
                    responses=[{"status": "pending"}, {"status": "shipped {{order_id}}"}])
    results = _call_tool_in_one_run(validator, [mock], ["A1", "A1"])
    assert results == [{"status": "pending"}, {"status": "shipped A1"}]
    assert real_calls == []


def test_a_mocked_function_tool_answers_from_its_handler(validator):
    def lookup(args, state):
        return {"status": "shipped" if args["order_id"].startswith("S") else "pending"}

    mock = MockTool(name="get_order_status", type="tool.adk", handler=lookup)
    assert _call_tool_in_one_run(validator, [mock], ["S9", "P1"]) == [{"status": "shipped"}, {"status": "pending"}]
    assert real_calls == []
