"""Unit tests for FluentTestCase.from_spans()."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from monocle_test_tools.span_loader import JSONSpanLoader
from monocle_test_tools.testcase import Agent, FluentTestCase, Tool

TRACE_FILE = Path(__file__).parent / "traces" / "trace1.json"


def _event(name, **attributes):
    return SimpleNamespace(name=name, attributes=attributes)


def _span(span_type, entity_1=None, entity_2=None, input=None, output=None,
          total_tokens=None, start_time=0, end_time=1, **attributes):
    attrs = {"span.type": span_type, **attributes}
    if entity_1 is not None:
        attrs["entity.1.name"] = entity_1
    if entity_2 is not None:
        attrs["entity.2.name"] = entity_2
    events = []
    if input is not None:
        events.append(_event("data.input", input=input))
    if output is not None:
        events.append(_event("data.output", response=output))
    if total_tokens is not None:
        events.append(_event("metadata", total_tokens=total_tokens))
    return SimpleNamespace(attributes=attrs, events=events,
                           start_time=start_time, end_time=end_time)


def test_from_spans_extracts_input_agents_tools_and_token_limit():
    spans = [
        _span("workflow", entity_1="my_app"),
        _span("agentic.turn", input="Book a flight to LAX", output="Booked"),
        _span("agentic.invocation", entity_1="supervisor",
              input="Book a flight to LAX", output="Booked"),
        _span("agentic.tool.invocation", entity_1="book_flight",
              entity_2="flight_agent", input="{'to': 'LAX'}", output="confirmed"),
        _span("inference", entity_2="gpt-4o", total_tokens=100),
        _span("inference", entity_2="gpt-4o", total_tokens=25),
    ]

    test_case = FluentTestCase.from_spans(spans)

    assert test_case.input == ("Book a flight to LAX",)
    assert test_case.agents == [
        Agent(name="supervisor", input="Book a flight to LAX", output="Booked")
    ]
    assert test_case.tools == [
        Tool(name="book_flight", input="{'to': 'LAX'}", output="confirmed",
             agent=Agent(name="flight_agent"))
    ]
    assert test_case.token_limit == 125
    assert test_case.name == "my_app"
    assert test_case.evals == []


def test_from_spans_explicit_name_and_evals_win():
    spans = [_span("workflow", entity_1="my_app"),
             _span("agentic.turn", input="hello")]

    test_case = FluentTestCase.from_spans(
        spans, name="my_test", evals=[{"name": "hallucinations", "result": "none"}])

    assert test_case.name == "my_test"
    assert [e.name for e in test_case.evals] == ["hallucinations"]
    assert [e.result for e in test_case.evals] == ["none"]


def test_from_spans_accepts_evals_in_the_keyed_shape():
    spans = [_span("agentic.turn", input="hello")]

    test_case = FluentTestCase.from_spans(spans, evals=[{"hallucinations": "none"}])

    assert [(e.name, e.result) for e in test_case.evals] == [("hallucinations", "none")]


def test_from_spans_dedupes_identical_calls_and_keeps_distinct_ones():
    spans = [
        _span("agentic.invocation", entity_1="agent_a", input="in", output="out"),
        _span("agentic.invocation", entity_1="agent_a", input="in", output="out"),
        _span("agentic.invocation", entity_1="agent_a", input="in2", output="out2"),
        _span("agentic.tool.invocation", entity_1="tool_a", entity_2="agent_a",
              input="t_in", output="t_out"),
        _span("agentic.tool.invocation", entity_1="tool_a", entity_2="agent_a",
              input="t_in", output="t_out"),
    ]

    test_case = FluentTestCase.from_spans(spans)

    assert [(a.name, a.input) for a in test_case.agents] == [
        ("agent_a", "in"), ("agent_a", "in2")]
    assert len(test_case.tools) == 1


def test_from_spans_orders_by_start_time_and_collects_every_turn_input():
    spans = [
        _span("agentic.turn", input="second turn", start_time=200, end_time=300),
        _span("agentic.turn", input="first turn", start_time=0, end_time=100),
        _span("agentic.invocation", entity_1="late_agent", start_time=250),
        _span("agentic.invocation", entity_1="early_agent", start_time=10),
    ]

    test_case = FluentTestCase.from_spans(spans)

    assert test_case.input == ("first turn", "second turn")
    assert [a.name for a in test_case.agents] == ["early_agent", "late_agent"]


def test_from_spans_falls_back_to_agentic_request_span_type():
    """Older traces tag the turn span as 'agentic.request'."""
    spans = [_span("agentic.request", input="Book a flight to LAX")]

    assert FluentTestCase.from_spans(spans).input == ("Book a flight to LAX",)


def test_from_spans_falls_back_to_agent_input_without_a_turn_span():
    spans = [_span("agentic.invocation", entity_1="solo_agent", input="do it")]

    assert FluentTestCase.from_spans(spans).input == ("do it",)


def test_from_spans_with_no_spans_returns_empty_test_case():
    test_case = FluentTestCase.from_spans([])

    assert test_case.input is None
    assert test_case.agents == []
    assert test_case.tools == []
    assert test_case.evals == []
    assert test_case.token_limit is None
    assert test_case.name == "monocle_test"


def test_from_spans_coerces_non_string_input_and_output():
    spans = [_span("agentic.invocation", entity_1="a",
                   input=["one", "two"], output={"k": "v"})]

    agent = FluentTestCase.from_spans(spans).agents[0]
    assert isinstance(agent.input, str)
    assert isinstance(agent.output, str)


def test_from_spans_on_recorded_trace():
    if not TRACE_FILE.exists():
        pytest.skip(f"Trace file {TRACE_FILE} not found")

    test_case = FluentTestCase.from_spans(JSONSpanLoader.from_json(str(TRACE_FILE)))

    assert test_case.name == "adk-travel-agent"
    assert test_case.input is not None and len(test_case.input) == 1
    assert "Book a flight from San Francisco to Mumbai" in test_case.input[0]
    assert {a.name for a in test_case.agents} == {
        "adk_flight_booking_agent_5", "adk_hotel_booking_agent_5",
        "adk_trip_summary_agent_5", "adk_supervisor_agent_5"}
    assert [t.name for t in test_case.tools] == ["adk_book_hotel_5"]
    assert test_case.tools[0].agent.name == "adk_hotel_booking_agent_5"
    # sum of total_tokens across the trace's four inference spans
    assert test_case.token_limit == 1204
