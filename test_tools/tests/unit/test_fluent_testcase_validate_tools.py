"""Unit tests for FluentTestCase.validate_tools()."""

from pathlib import Path

import pytest

from monocle_test_tools import TraceAssertion
from monocle_test_tools.span_loader import JSONSpanLoader
from monocle_test_tools.testcase import FluentTestCase

TRACE_FILE = Path(__file__).parent / "traces" / "trace1.json"

# The one tool invocation recorded in trace1.json.
TOOL = "adk_book_hotel_5"
CALLING_AGENT = "adk_hotel_booking_agent_5"
TOOL_INPUT = "{'city': 'Mumbai', 'hotel_name': 'Marriot Intercontinental'}"
TOOL_OUTPUT = ("{'status': 'success', 'message': 'Successfully booked a stay at "
               "Marriot Intercontinental in Mumbai.'}")


@pytest.fixture(autouse=True)
def _reset_trace_assertion_class_state():
    """Recorded assertions live on the class, so clear them around every test."""
    TraceAssertion._assertion_errors = []
    yield
    TraceAssertion._assertion_errors = []


@pytest.fixture(name="asserter")
def asserter_fixture():
    """A trace asserter holding trace1.json.

    ``load_spans`` puts the spans on the process-wide validator, so the teardown
    cleanup matters: without it the loaded spans leak into later test modules.
    """
    asserter = TraceAssertion.get_trace_asserter()
    asserter.load_spans(JSONSpanLoader.from_json(str(TRACE_FILE)))
    yield asserter
    asserter.cleanup()


def _drain_failures() -> list[str]:
    """Recorded failure messages, cleared so they don't flip this test's own outcome."""
    messages = [assertion["message"] for assertion in TraceAssertion._assertion_errors]
    TraceAssertion._assertion_errors = []
    return messages


def test_validate_tools_passes_for_a_matching_tool(asserter):
    test_case = FluentTestCase.model_validate({
        "tools": [{TOOL: {"input": TOOL_INPUT, "output": TOOL_OUTPUT,
                          "agent": {CALLING_AGENT: {}}}}]})

    test_case.validate_tools(asserter)

    assert _drain_failures() == []


def test_validate_tools_passes_for_a_name_only_tool(asserter):
    test_case = FluentTestCase.model_validate({"tools": [{TOOL: {}}]})

    test_case.validate_tools(asserter)

    assert _drain_failures() == []


def test_validate_tools_records_a_failure_for_a_tool_that_was_not_called(asserter):
    test_case = FluentTestCase.model_validate({"tools": [{"book_train": {}}]})

    test_case.validate_tools(asserter)

    failures = _drain_failures()
    assert len(failures) == 1
    assert "book_train" in failures[0]


def test_validate_tools_records_a_failure_for_the_wrong_calling_agent(asserter):
    test_case = FluentTestCase.model_validate({
        "tools": [{TOOL: {"agent": {"adk_flight_booking_agent_5": {}}}}]})

    test_case.validate_tools(asserter)

    failures = _drain_failures()
    assert len(failures) == 1
    assert "adk_flight_booking_agent_5" in failures[0]


def test_validate_tools_records_a_failure_for_the_wrong_output(asserter):
    test_case = FluentTestCase.model_validate({
        "tools": [{TOOL: {"output": "no rooms available"}}]})

    test_case.validate_tools(asserter)

    failures = _drain_failures()
    assert len(failures) == 1
    assert "no rooms available" in failures[0]


def test_validate_tools_checks_every_tool_independently(asserter):
    """A failing tool must not narrow the spans the next tool is looked for in."""
    test_case = FluentTestCase.model_validate({
        "tools": [{"book_train": {}}, {TOOL: {"output": TOOL_OUTPUT}}]})

    test_case.validate_tools(asserter)

    failures = _drain_failures()
    assert len(failures) == 1
    assert "book_train" in failures[0]


def test_validate_tools_with_no_tools_is_a_no_op(asserter):
    FluentTestCase().validate_tools(asserter)

    assert _drain_failures() == []


def test_validate_tools_returns_the_test_case_for_chaining(asserter):
    test_case = FluentTestCase.model_validate({"tools": [{TOOL: {}}]})

    assert test_case.validate_tools(asserter) is test_case
    _drain_failures()
