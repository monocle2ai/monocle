"""The same span added twice is held once.

An agent can both return its spans with its answer and export them, so a test
reading both routes sees each span twice -- which would double every count.
"""
import os

import pytest

from monocle_test_tools.span_loader import JSONSpanLoader
from monocle_test_tools.validator import MonocleValidator

TRACE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "traces", "trace1.json")


@pytest.fixture(name="validator")
def validator_fixture():
    validator = MonocleValidator()
    validator.cleanup()
    yield validator
    validator.cleanup()


def test_the_same_spans_twice_are_held_once(validator):
    spans = JSONSpanLoader.from_json(TRACE)

    validator.add_remote_spans(spans)
    validator.add_remote_spans(spans)

    assert len(validator.spans) == len(spans)


def test_spans_from_another_trace_are_all_kept(validator):
    spans = JSONSpanLoader.from_json(TRACE)
    half = len(spans) // 2

    validator.add_remote_spans(spans[:half])
    validator.add_remote_spans(spans)          # the first half repeats, the rest is new

    assert len(validator.spans) == len(spans)


def test_adding_nothing_changes_nothing(validator):
    spans = JSONSpanLoader.from_json(TRACE)
    validator.add_remote_spans(spans)

    validator.add_remote_spans([])

    assert len(validator.spans) == len(spans)
