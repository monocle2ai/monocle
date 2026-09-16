"""Shared testcase-argument resolution for the fluent entry points."""
import pytest

from monocle_test_tools.testcase import Eval, FluentTestCase
from monocle_test_tools.testcase_args import resolve_testcase


def test_dict_is_converted_to_a_model():
    tc = resolve_testcase({"evals": {"hallucination": "minor"}})

    assert isinstance(tc, FluentTestCase)
    assert tc.evals == [Eval(name="hallucination", result="minor")]


def test_model_is_returned_unchanged():
    original = FluentTestCase(input=("go",))

    assert resolve_testcase(original) is original


def test_conflicting_argument_raises_naming_it():
    with pytest.raises(ValueError, match="'eval_name' cannot be combined with 'testcase'"):
        resolve_testcase({"evals": {"a": "x"}}, eval_name="hallucination")


def test_all_conflicting_arguments_are_named():
    with pytest.raises(ValueError, match=r"'eval_name', 'expected'"):
        resolve_testcase({"evals": {"a": "x"}}, eval_name="h", expected="good")


def test_none_valued_arguments_are_not_conflicts():
    tc = resolve_testcase({"evals": {"a": "x"}}, eval_name=None, expected=None)

    assert tc.evals == [Eval(name="a", result="x")]


def test_empty_positional_args_tuple_is_not_a_conflict():
    """run_agent passes its *args through; an empty tuple means none were given."""
    tc = resolve_testcase({"input": "go"}, args=())

    assert tc.input == ("go",)


def test_non_empty_positional_args_tuple_is_a_conflict():
    with pytest.raises(ValueError, match="'args' cannot be combined with 'testcase'"):
        resolve_testcase({"input": "go"}, args=("go",))


def test_wrong_type_raises():
    with pytest.raises(TypeError, match="testcase must be a FluentTestCase or dict"):
        resolve_testcase("not a testcase")


import os

from monocle_test_tools.schema import FactID
from monocle_test_tools.span_loader import JSONSpanLoader
from monocle_test_tools.testcase_args import (factid_import_kwargs,
                                              turn_inputs_from_spans)

TRACES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "traces")
TRACE1_TURN_INPUT = ("Book a flight from San Francisco to Mumbai for 26th Nov 2025. "
                     "Book a two queen room at Marriot Intercontinental at Juhu, "
                     "Mumbai for 27th Nov 2025 for 4 nights.")
TRACE3_TURN_INPUT = "What happened in Clippers game on 22 Nov 2025"


class TestFactIdImportKwargs:

    def test_default_traces_fact_name_maps_to_trace(self):
        """FactID defaults to "traces"; import_traces spells the same thing "trace"."""
        kwargs = factid_import_kwargs(FactID(fact_id="t1"))

        assert kwargs == {"trace_source": "file", "id": "t1", "fact_name": "trace"}

    def test_trace_fact_name_passes_through(self):
        kwargs = factid_import_kwargs(FactID(fact_id="t1", fact_name="trace",
                                             source="okahu"))

        assert kwargs == {"trace_source": "okahu", "id": "t1", "fact_name": "trace"}

    def test_session_fact_name_passes_through(self):
        kwargs = factid_import_kwargs(FactID(fact_id="s1", fact_name="session",
                                             source="okahu"))

        assert kwargs == {"trace_source": "okahu", "id": "s1", "fact_name": "session"}

    def test_unrecognized_fact_name_becomes_a_custom_scope(self):
        kwargs = factid_import_kwargs(FactID(fact_id="t_123", fact_name="test_runid",
                                             source="okahu"))

        assert kwargs == {"trace_source": "okahu", "id": "t_123",
                          "fact_name": "scope", "scope_name": "test_runid"}

    def test_missing_fact_id_raises(self):
        with pytest.raises(ValueError, match="fact_id is required"):
            factid_import_kwargs(FactID(fact_name="trace"))


class TestTurnInputsFromSpans:

    def test_modern_turn_span(self):
        spans = JSONSpanLoader.from_json(os.path.join(TRACES_DIR, "trace3.json"))

        assert turn_inputs_from_spans(spans) == (TRACE3_TURN_INPUT,)

    def test_legacy_agentic_request_span(self):
        """Traces recorded by older Monocle tag the turn span "agentic.request"."""
        spans = JSONSpanLoader.from_json(os.path.join(TRACES_DIR, "trace1.json"))

        assert turn_inputs_from_spans(spans) == (TRACE1_TURN_INPUT,)

    def test_span_order_follows_start_time_not_list_order(self):
        spans = JSONSpanLoader.from_json(os.path.join(TRACES_DIR, "trace3.json"))

        assert turn_inputs_from_spans(list(reversed(spans))) == (TRACE3_TURN_INPUT,)

    def test_falls_back_to_first_agent_invocation(self):
        """A partial trace with no turn span still yields what the agent was asked."""
        spans = [s for s in JSONSpanLoader.from_json(
            os.path.join(TRACES_DIR, "trace3.json"))
            if (s.attributes or {}).get("span.type") != "agentic.turn"]

        inputs = turn_inputs_from_spans(spans)

        assert len(inputs) == 1
        assert inputs[0]

    def test_no_input_available_raises(self):
        with pytest.raises(ValueError, match="no turn or agent invocation input"):
            turn_inputs_from_spans([])


class TestListValuedInput:
    """A span's data.input is often a list, not a string.

    22 of the 40 data.input events in the sample traces hold a list -- usually
    the JSON message strings an agent was invoked with. Passing that list
    through would hand run_agent a list where a prompt belongs.
    """

    def _span(self, value, span_type="agentic.turn"):
        from unittest.mock import MagicMock
        event = MagicMock()
        event.name = "data.input"
        event.attributes = {"input": value}
        span = MagicMock()
        span.attributes = {"span.type": span_type}
        span.events = [event]
        span.start_time = 1
        return span

    def test_list_of_strings_joins_with_newlines(self):
        spans = [self._span(['{"system": "You are..."}', '{"user": "Book a flight"}'])]

        assert turn_inputs_from_spans(spans) == (
            '{"system": "You are..."}\n{"user": "Book a flight"}',)

    def test_list_of_dicts_is_stringified_per_element(self):
        spans = [self._span([{"role": "user", "content": "Book a flight"}])]

        assert turn_inputs_from_spans(spans) == (
            "{'role': 'user', 'content': 'Book a flight'}",)

    def test_a_plain_string_is_untouched(self):
        spans = [self._span("Book a flight")]

        assert turn_inputs_from_spans(spans) == ("Book a flight",)

    def test_an_empty_list_is_not_an_input(self):
        """[] carries no prompt, so it must not become the empty string."""
        with pytest.raises(ValueError, match="no turn or agent invocation input"):
            turn_inputs_from_spans([self._span([])])

    def test_empty_elements_are_dropped(self):
        spans = [self._span(["", "Book a flight", None])]

        assert turn_inputs_from_spans(spans) == ("Book a flight",)

    def test_the_agent_invocation_fallback_converts_too(self):
        spans = [self._span(['{"user": "Book a flight"}'],
                            span_type="agentic.invocation")]

        assert turn_inputs_from_spans(spans) == ('{"user": "Book a flight"}',)
