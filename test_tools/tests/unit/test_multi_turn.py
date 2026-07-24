"""Unit tests for the multi-turn test support (MultiTurnTestCase + validator).

These tests exercise the parts of the multi-turn feature that do not need a
live agent: the MultiTurnTestCase schema, the turn-to-turn input chaining
helper, and session-level validation against pre-loaded spans. The end-to-end
run against a live agent session is covered by the integration tests.
"""
import os
import uuid

import pytest

from monocle_test_tools import (
    MonocleValidator, MultiTurnTestCase, TestCase, TestSpan, Entity
)
from monocle_test_tools.constants import SESSION_SCOPE_NAME
from span_loader import JSONSpanLoader


@pytest.fixture(scope="function")
def setup():
    """MonocleValidator preloaded with a saved trace's spans."""
    validator = MonocleValidator()
    current_script_path = os.path.abspath(__file__)
    spans = JSONSpanLoader.from_json(
        os.path.join(os.path.dirname(current_script_path), "traces/trace1.json")
    )
    validator.memory_exporter.export(spans)
    yield validator
    validator.memory_exporter.clear()
    validator.cleanup()


def test_multi_turn_schema_from_dict():
    """A MultiTurnTestCase can be built from a plain dict of turns."""
    mtc = MultiTurnTestCase.model_validate({
        "turns": [
            {"test_input": ["Book a flight to Mumbai"]},
            {"test_input": ["Yes, please proceed"]},
        ],
        "session_output": "booked",
    })
    assert len(mtc.turns) == 2
    assert all(isinstance(t, TestCase) for t in mtc.turns)
    assert mtc.session_output == "booked"


def test_multi_turn_schema_requires_turns():
    """A MultiTurnTestCase with no turns is rejected."""
    with pytest.raises(ValueError):
        MultiTurnTestCase(turns=[])


def test_multi_turn_session_id_defaults_none():
    """session_id is optional and defaults to None (assigned at run time)."""
    mtc = MultiTurnTestCase(turns=[TestCase(test_input=("hi",))])
    assert mtc.session_id is None


def test_chain_turn_input_substitutes_placeholder():
    chained = MonocleValidator._chain_turn_input(
        ("Use {previous_output} as the city",), "Mumbai"
    )
    assert chained == ("Use Mumbai as the city",)


def test_chain_turn_input_no_placeholder_unchanged():
    chained = MonocleValidator._chain_turn_input(("plain input",), "Mumbai")
    assert chained == ("plain input",)


def test_chain_turn_input_none_previous_output():
    chained = MonocleValidator._chain_turn_input(("{previous_output}",), None)
    assert chained == ("{previous_output}",)


def test_session_spans_validate_against_accumulated(setup):
    """session_spans assertions run against the accumulated session spans.

    Simulate the post-run state: _test_all_up_spans holds every turn's spans,
    and _spans is pointed at that accumulation for session validation.
    """
    spans = setup.memory_exporter.get_finished_spans()
    setup._test_all_up_spans = tuple(spans)
    setup._spans = tuple(spans)

    session_case = TestCase(
        test_spans=[
            TestSpan(
                span_type="agentic.tool.invocation",
                entities=[
                    Entity(type="tool", name="adk_book_hotel_5"),
                    Entity(type="agent", name="adk_hotel_booking_agent_5"),
                ],
            )
        ]
    )
    assert setup.validate(session_case)


def test_session_spans_negative(setup):
    """A tool never invoked across the session fails a positive assertion."""
    spans = setup.memory_exporter.get_finished_spans()
    setup._spans = tuple(spans)

    session_case = TestCase(
        test_spans=[
            TestSpan(
                span_type="agentic.tool.invocation",
                entities=[
                    Entity(type="tool", name="nonexistent_tool"),
                    Entity(type="agent", name="adk_hotel_booking_agent_5"),
                ],
                test_type="negative",
            )
        ]
    )
    assert setup.validate(session_case)


if __name__ == "__main__":
    pytest.main([__file__])
