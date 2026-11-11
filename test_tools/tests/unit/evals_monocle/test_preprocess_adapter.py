import json
from pathlib import Path

from opentelemetry.sdk.trace import Event, ReadableSpan
from opentelemetry.trace import SpanContext, TraceFlags

from monocle_test_tools.evals.monocle.preprocess import _iso_time_of, to_trace_file_shape
from monocle_test_tools.file_span_loader import JSONSpanLoader

REFUND_TRACE = Path(__file__).parents[2] / "integration" / "json_loader_traces" / "642dbd9d0dfcfdbdc8849f67f34c8a19.json"
ONE_SECOND_IN_NANOSECONDS = 1_000_000_000


def span_with(attributes: dict) -> ReadableSpan:
    context = SpanContext(trace_id=0xA1, span_id=0xB1, is_remote=False, trace_flags=TraceFlags(1))
    return ReadableSpan(
        name="step",
        context=context,
        attributes=attributes,
        start_time=ONE_SECOND_IN_NANOSECONDS,
        end_time=2 * ONE_SECOND_IN_NANOSECONDS,
    )


def adapted_attributes(attributes: dict) -> dict:
    return to_trace_file_shape([span_with(attributes)])["spans"][0]["attributes"]


def test_entity_keys_fold_into_list_in_order():
    attributes = {
        "entity.1.name": "first",
        "entity.1.type": "workflow.langgraph",
        "entity.2.name": "second",
        "entity.2.type": "app_hosting.generic",
        "entity.count": 2,
        "span.type": "workflow",
    }

    assert adapted_attributes(attributes) == {
        "entity": [
            {"name": "first", "type": "workflow.langgraph"},
            {"name": "second", "type": "app_hosting.generic"},
        ],
        "entity.count": 2,
        "span.type": "workflow",
    }


def test_span_without_entities_keeps_attributes():
    attributes = {"span.type": "inference", "scope.agentic.turn": "t1"}

    assert adapted_attributes(attributes) == attributes


def with_entity_count_the_loader_adds(fixture_attributes: dict) -> dict:
    if "entity" in fixture_attributes and "entity.count" not in fixture_attributes:
        return {**fixture_attributes, "entity.count": len(fixture_attributes["entity"])}
    return fixture_attributes


def test_round_trip_of_fixture_attributes():
    fixture = json.loads(REFUND_TRACE.read_text())
    loaded_spans = JSONSpanLoader.from_json_str(json.dumps(fixture["spans"]))

    adapted = to_trace_file_shape(loaded_spans)

    assert adapted["trace_id"] == fixture["trace_id"]
    assert len(adapted["spans"]) == len(fixture["spans"])
    for adapted_span, fixture_span in zip(adapted["spans"], fixture["spans"]):
        assert adapted_span["attributes"] == with_entity_count_the_loader_adds(fixture_span["attributes"])
        assert adapted_span["events"] == fixture_span["events"]


def test_times_are_iso_strings():
    adapted_span = to_trace_file_shape([span_with({})])["spans"][0]

    assert adapted_span["start_time"] == "1970-01-01T00:00:01.000000Z"
    assert adapted_span["end_time"] == "1970-01-01T00:00:02.000000Z"


def test_time_just_below_the_next_second_carries_into_it():
    assert _iso_time_of(1_999_999_700) == "1970-01-01T00:00:02.000000Z"


def test_tuple_values_become_lists():
    context = SpanContext(trace_id=0xA1, span_id=0xB1, is_remote=False, trace_flags=TraceFlags(1))
    span = ReadableSpan(
        name="step",
        context=context,
        attributes={"tags": ("a", "b"), "entity.1.roles": ("x", "y")},
        events=[Event(name="data.input", attributes={"input": ("hello",)}, timestamp=ONE_SECOND_IN_NANOSECONDS)],
    )

    adapted_span = to_trace_file_shape([span])["spans"][0]

    assert adapted_span["attributes"] == {"tags": ["a", "b"], "entity": [{"roles": ["x", "y"]}]}
    assert adapted_span["events"][0]["attributes"] == {"input": ["hello"]}


def test_entity_key_without_a_field_stays_a_plain_attribute():
    assert adapted_attributes({"entity.1": "x"}) == {"entity.1": "x"}
