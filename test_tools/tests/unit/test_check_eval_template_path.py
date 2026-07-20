import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock
from monocle_test_tools.fluent_api import TraceAssertion


INNER_TEMPLATE = {
    "name": "test_template",
    "eval_prompt": "x",
    "structure_output": {
        "label": {"enums": ["a", "b"], "description": "x"},
        "explanation": {"description": "x"},
    },
}
WRAPPED_TEMPLATE = {"template": INNER_TEMPLATE}


def _make_asserter(eval_result=("a", "ok")):
    """Build a TraceAssertion with one span and a mocked evaluator."""
    span = MagicMock()
    eval_mock = MagicMock()
    eval_mock.evaluate.return_value = eval_result
    return TraceAssertion(filtered_spans=[span], _eval=eval_mock)


class TestCheckEvalTemplatePath:

    def test_wrapped_template_file_is_unwrapped_before_evaluate(self, tmp_path):
        """File on disk has {"template": {...inner...}} (matches API request body
        shape). check_eval must unwrap so evaluate() receives just the inner dict."""
        path = tmp_path / "tpl.json"
        path.write_text(json.dumps(WRAPPED_TEMPLATE), encoding="utf-8")
        asserter = _make_asserter(eval_result=("a", "ok"))

        result = asserter.check_eval(template_path=str(path), expected="a")

        assert not result.has_assertions(), result.get_assertion_messages()
        asserter._eval.evaluate.assert_called_once()
        _, kwargs = asserter._eval.evaluate.call_args
        assert kwargs["template"] == INNER_TEMPLATE  # unwrapped
        assert kwargs["fact_name"] == "traces"

    def test_unwrapped_template_file_passes_through(self, tmp_path):
        """File on disk is just the inner template (no outer wrapper).
        check_eval passes it through unchanged."""
        path = tmp_path / "tpl.json"
        path.write_text(json.dumps(INNER_TEMPLATE), encoding="utf-8")
        asserter = _make_asserter(eval_result=("a", "ok"))

        result = asserter.check_eval(template_path=str(path), expected="a")

        assert not result.has_assertions(), result.get_assertion_messages()
        _, kwargs = asserter._eval.evaluate.call_args
        assert kwargs["template"] == INNER_TEMPLATE

    def test_missing_file_raises_clean_assertion(self, tmp_path):
        missing = tmp_path / "does_not_exist.json"
        asserter = _make_asserter()

        result = asserter.check_eval(template_path=str(missing), expected="a")

        assert result.has_assertions()
        msg = result.get_assertion_messages()
        assert "Custom template file not found" in msg
        assert str(missing) in msg
        # Clear assertions so pytest hook doesn't fail the test
        TraceAssertion._assertion_errors = []

    def test_invalid_json_raises_clean_assertion(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not valid json", encoding="utf-8")
        asserter = _make_asserter()

        result = asserter.check_eval(template_path=str(path), expected="a")

        assert result.has_assertions()
        msg = result.get_assertion_messages()
        assert "Custom template file is not valid JSON" in msg
        assert str(path) in msg
        # Clear assertions so pytest hook doesn't fail the test
        TraceAssertion._assertion_errors = []

    def test_both_eval_name_and_template_path_raises_value_error(self, tmp_path):
        path = tmp_path / "tpl.json"
        path.write_text(json.dumps(WRAPPED_TEMPLATE), encoding="utf-8")
        asserter = _make_asserter()

        with pytest.raises(ValueError, match="exactly one"):
            asserter.check_eval(
                eval_name="hallucination",
                template_path=str(path),
                expected="a",
            )

    def test_neither_eval_name_nor_template_path_raises_value_error(self):
        asserter = _make_asserter()

        with pytest.raises(ValueError, match="exactly one"):
            asserter.check_eval(expected="a")

    def test_eval_name_and_template_dict_both_raises_value_error(self):
        """Passing both eval_name and template (dict) selectors is rejected,
        matching the eval_name/template_path exactly-one behavior."""
        asserter = _make_asserter()

        with pytest.raises(ValueError, match="exactly one"):
            asserter.check_eval(
                eval_name="hallucination",
                template=dict(INNER_TEMPLATE),
                expected="a",
            )

    def test_template_dict_selector_reaches_evaluate_unmodified(self):
        """Regression test: an inline `template` dict passed to check_eval in
        span mode must reach self._eval.evaluate(..., template=...) as-is,
        not be clobbered by a stray `template = None` reset."""
        asserter = _make_asserter(eval_result=("a", "ok"))
        template_dict = dict(INNER_TEMPLATE)

        result = asserter.check_eval(template=template_dict, expected="a")

        assert not result.has_assertions(), result.get_assertion_messages()
        _, kwargs = asserter._eval.evaluate.call_args
        assert kwargs["template"] == template_dict
        assert kwargs["template"] is not None
        # eval_name should be derived from the template's "name" field.
        assert kwargs["eval_name"] == "test_template"
