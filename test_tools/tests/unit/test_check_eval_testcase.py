"""check_eval driven by a FluentTestCase, and the chain-mixing guard.

A test case can carry several evals, so one call runs several. They must all be
graded and all reported: record_assertion keeps only the first AssertionError of
a chain, so failures are accumulated into one.
"""
import json

from unittest.mock import MagicMock

import pytest

from monocle_test_tools.fluent_api import TraceAssertion


@pytest.fixture(autouse=True)
def _reset_trace_assertion_class_state():
    TraceAssertion._assertion_errors = []
    TraceAssertion._eval_report = None
    TraceAssertion._eval_stashes = []
    TraceAssertion._okahu_filter = None
    yield
    TraceAssertion._assertion_errors = []
    TraceAssertion._eval_report = None
    TraceAssertion._eval_stashes = []
    TraceAssertion._okahu_filter = None


def _asserter(labels):
    """An asserter whose evaluator returns `labels[eval_name]`."""
    eval_mock = MagicMock()
    eval_mock.last_fact_results = None

    def evaluate(filtered_spans=None, eval_name=None, fact_name=None, template=None):
        key = eval_name if eval_name in labels else "__default__"
        return labels[key], "because"

    eval_mock.evaluate.side_effect = evaluate
    return TraceAssertion(filtered_spans=[MagicMock()], _eval=eval_mock)


def _drain():
    messages = [a["message"] for a in TraceAssertion._assertion_errors]
    TraceAssertion._assertion_errors = []
    return messages


def test_single_eval_passes():
    asserter = _asserter({"hallucination": "minor_hallucination"})

    asserter.check_eval(testcase={"expected": {
        "evals": {"hallucination": "minor_hallucination"}}})

    assert _drain() == []


def test_all_evals_are_run():
    asserter = _asserter({"hallucination": "minor", "frustration": "none"})

    asserter.check_eval(testcase={"evals": {"hallucination": "minor",
                                            "frustration": "none"}})

    assert _drain() == []
    assert len(TraceAssertion._eval_stashes) == 2


def test_every_failing_eval_is_reported():
    asserter = _asserter({"hallucination": "major", "frustration": "high"})

    result = asserter.check_eval(testcase={"evals": {"hallucination": "minor",
                                                     "frustration": "none"}})

    assert result.has_assertions()
    message = result.get_assertion_messages()
    assert "hallucination" in message
    assert "frustration" in message
    _drain()


def test_a_path_eval_name_routes_to_a_template(tmp_path):
    template = tmp_path / "my_eval.json"
    template.write_text('{"name": "my_eval", "eval_prompt": "p"}', encoding="utf-8")
    asserter = _asserter({"my_eval": "good", "__default__": "good"})

    asserter.check_eval(testcase={"evals": [{str(template): "good"}]})

    assert _drain() == []


def test_empty_evals_raises():
    with pytest.raises(ValueError, match="no evals to check"):
        _asserter({}).check_eval(testcase={"input": "go"})


@pytest.mark.parametrize("conflicting", [
    {"eval_name": "hallucination"}, {"expected": "good"},
    {"not_expected": "bad"}, {"template_path": "x.json"},
    {"template": {"name": "t"}},
])
def test_conflicting_arguments_raise(conflicting):
    asserter = _asserter({"hallucination": "minor"})

    with pytest.raises(ValueError, match="cannot be combined with 'testcase'"):
        asserter.check_eval(testcase={"evals": {"hallucination": "minor"}},
                            **conflicting)


def test_filter_mode_with_testcase_raises():
    asserter = _asserter({"hallucination": "minor"})
    asserter._okahu_filter = {"workflows": ["wf"], "start_time": "a",
                              "end_time": "b", "fact_name": "traces"}

    with pytest.raises(ValueError, match="time-window"):
        asserter.check_eval(testcase={"evals": {"hallucination": "minor"}})


class TestChainMixingGuard:

    def test_either_form_is_allowed_at_chain_start(self):
        asserter = _asserter({"hallucination": "minor"})

        asserter.check_eval(eval_name="hallucination", expected="minor")
        asserter.check_eval(testcase={"evals": {"hallucination": "minor"}})

        assert _drain() == []

    def test_omitting_testcase_mid_chain_raises(self):
        asserter = _asserter({"hallucination": "minor"})
        scope = asserter.check_eval(testcase={"evals": {"hallucination": "minor"}})

        with pytest.raises(ValueError, match="testcase"):
            scope.called_tool("book_flight")

    def test_adding_testcase_mid_chain_raises(self):
        asserter = _asserter({"hallucination": "minor"})
        scope = asserter.called_agent("supervisor")
        _drain()

        with pytest.raises(ValueError, match="testcase"):
            scope.check_eval(testcase={"evals": {"hallucination": "minor"}})

    def test_a_fresh_chain_is_unaffected(self):
        """The mode lives on the derived asserter, not the fixture's own."""
        asserter = _asserter({"hallucination": "minor"})
        asserter.check_eval(testcase={"evals": {"hallucination": "minor"}})

        asserter.check_eval(eval_name="hallucination", expected="minor")

        assert _drain() == []


class TestEvalNameClassification:
    """A test-case eval name is classified the way check_eval classifies its own.

    A test case usually arrives as JSON, which has no Path type, so a custom
    template reaches us as a *string* path. Treating only pathlib.Path as a
    template sent that string on as a built-in template name, and the eval
    service was asked for a template called "./evals/my_eval.json".
    """

    TEMPLATE = {"name": "my_eval", "eval_prompt": "p"}

    def _template_file(self, tmp_path):
        path = tmp_path / "my_eval.json"
        path.write_text(json.dumps(self.TEMPLATE), encoding="utf-8")
        return path

    def _spy(self):
        """An asserter recording exactly what evaluate() was handed."""
        eval_mock = MagicMock()
        eval_mock.last_fact_results = None
        eval_mock.evaluate.return_value = ("good", "because")
        return TraceAssertion(filtered_spans=[MagicMock()], _eval=eval_mock), eval_mock

    def test_a_string_path_is_loaded_as_a_template(self, tmp_path):
        """The regression: a str path must route to template_path, not eval_name."""
        path = self._template_file(tmp_path)
        asserter, eval_mock = self._spy()

        asserter.check_eval(testcase={"evals": [{str(path): "good"}]})

        assert _drain() == []
        kwargs = eval_mock.evaluate.call_args.kwargs
        assert kwargs["template"] == self.TEMPLATE
        assert kwargs["eval_name"] == "my_eval"

    def test_a_path_object_still_works(self, tmp_path):
        path = self._template_file(tmp_path)
        asserter, eval_mock = self._spy()

        asserter.check_eval(testcase={"evals": [{path: "good"}]})

        assert _drain() == []
        assert eval_mock.evaluate.call_args.kwargs["template"] == self.TEMPLATE

    def test_a_bare_name_stays_a_builtin(self, tmp_path):
        asserter, eval_mock = self._spy()

        asserter.check_eval(testcase={"evals": {"hallucination": "good"}})

        assert _drain() == []
        kwargs = eval_mock.evaluate.call_args.kwargs
        assert kwargs["eval_name"] == "hallucination"
        assert kwargs["template"] is None

    def test_a_relative_path_without_json_suffix_is_a_template(self, tmp_path):
        """classify_eval_input keys off ./ and separators too, not just .json."""
        nested = tmp_path / "evals"
        nested.mkdir()
        path = nested / "my_eval.json"
        path.write_text(json.dumps(self.TEMPLATE), encoding="utf-8")
        asserter, eval_mock = self._spy()

        asserter.check_eval(testcase={"evals": [{str(path): "good"}]})

        assert _drain() == []
        assert eval_mock.evaluate.call_args.kwargs["template"] == self.TEMPLATE

    def test_a_missing_template_file_is_reported(self, tmp_path):
        asserter, _ = self._spy()

        asserter.check_eval(
            testcase={"evals": [{str(tmp_path / "nope.json"): "good"}]})

        messages = _drain()
        assert len(messages) == 1
        assert "not found" in messages[0].lower()
