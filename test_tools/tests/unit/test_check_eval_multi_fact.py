"""check_eval must grade *every* fact before raising.

When fact_name resolves to many facts (one per turn of a multi-turn session),
bailing out on the first failing fact hides the rest -- a three-bad-turn run
should report all three, the way the filtered path aggregates its failing
scenarios in _check_eval_filtered.
"""

import pytest
from unittest.mock import MagicMock
from monocle_test_tools.fluent_api import TraceAssertion


def _make_asserter(fact_results, returned=("good", "ok")):
    """TraceAssertion with one span and an evaluator returning N per-fact results."""
    eval_mock = MagicMock()
    eval_mock.evaluate.return_value = returned
    eval_mock.last_fact_results = [
        {"fact_id": fact_id, "eval_result": {"label": label, "explanation": explanation}}
        for fact_id, label, explanation in fact_results
    ]
    return TraceAssertion(filtered_spans=[MagicMock()], _eval=eval_mock)


@pytest.fixture(autouse=True)
def _reset_trace_assertion_class_state():
    """Reset shared class-level TraceAssertion state before AND after each test.

    These tests build a TraceAssertion directly (bypassing the
    monocle_trace_asserter fixture and its cleanup()) and intentionally leave a
    recorded assertion behind; without this the dirty class-level
    _assertion_errors would flip the next test to failed via pytest_plugin.py's
    pytest_runtest_makereport hook.
    """
    TraceAssertion._assertion_errors = []
    TraceAssertion._eval_report = None
    TraceAssertion._okahu_filter = None
    yield
    TraceAssertion._assertion_errors = []
    TraceAssertion._eval_report = None
    TraceAssertion._okahu_filter = None


class TestCheckEvalMultiFact:

    def test_every_failing_fact_is_reported(self):
        """Three bad turns -> all three fact ids in the single recorded failure."""
        asserter = _make_asserter([
            ("turn-1", "bad", "first turn drifted"),
            ("turn-2", "bad", "second turn drifted"),
            ("turn-3", "bad", "third turn drifted"),
        ])

        try:
            result = asserter.check_eval(eval_name="hallucination", expected="good")

            assert result.has_assertions()
            msg = result.get_assertion_messages()
            assert "turn-1" in msg
            assert "turn-2" in msg
            assert "turn-3" in msg
            assert "failed for 3 of 3 facts" in msg
            # Per-fact explanations survive the aggregation.
            assert "first turn drifted" in msg
            assert "third turn drifted" in msg
        finally:
            TraceAssertion._assertion_errors = []

    def test_passing_facts_are_not_reported(self):
        """A pass between two failures is graded, not reported."""
        asserter = _make_asserter([
            ("turn-1", "bad", "drifted"),
            ("turn-2", "good", "fine"),
            ("turn-3", "bad", "drifted again"),
        ])

        try:
            result = asserter.check_eval(eval_name="hallucination", expected="good")

            assert result.has_assertions()
            msg = result.get_assertion_messages()
            assert "failed for 2 of 3 facts" in msg
            assert "turn-1" in msg
            assert "turn-3" in msg
            assert "turn-2" not in msg
        finally:
            TraceAssertion._assertion_errors = []

    def test_not_expected_failures_are_aggregated(self):
        """The not_expected branch aggregates the same way as expected."""
        asserter = _make_asserter([
            ("turn-1", "major_hallucination", "made things up"),
            ("turn-2", "major_hallucination", "made more things up"),
        ])

        try:
            result = asserter.check_eval(
                eval_name="hallucination", not_expected="major_hallucination")

            assert result.has_assertions()
            msg = result.get_assertion_messages()
            assert "failed for 2 of 2 facts" in msg
            assert "turn-1" in msg
            assert "turn-2" in msg
            assert "matched an unexpected result" in msg
        finally:
            TraceAssertion._assertion_errors = []

    def test_single_failing_fact_keeps_plain_message(self):
        """One failure reads as before -- no aggregate header wrapped around it."""
        asserter = _make_asserter([
            ("turn-1", "good", "fine"),
            ("turn-2", "bad", "drifted"),
        ])

        try:
            result = asserter.check_eval(eval_name="hallucination", expected="good")

            assert result.has_assertions()
            msg = result.get_assertion_messages()
            assert "facts:" not in msg
            assert "did not match expected result for fact 'turn-2'" in msg
        finally:
            TraceAssertion._assertion_errors = []

    def test_custom_message_overrides_aggregate(self):
        """An explicit message replaces the whole per-fact breakdown."""
        asserter = _make_asserter([
            ("turn-1", "bad", "drifted"),
            ("turn-2", "bad", "drifted again"),
        ])

        try:
            result = asserter.check_eval(
                eval_name="hallucination", expected="good", message="turns went bad")

            assert result.has_assertions()
            msg = result.get_assertion_messages()
            assert "turns went bad" in msg
            assert "turn-1" not in msg
        finally:
            TraceAssertion._assertion_errors = []

    def test_all_facts_passing_records_nothing(self):
        asserter = _make_asserter([
            ("turn-1", "good", "fine"),
            ("turn-2", "good", "also fine"),
        ])

        result = asserter.check_eval(eval_name="hallucination", expected="good")

        assert not result.has_assertions(), result.get_assertion_messages()
