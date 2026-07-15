import pytest
import requests
from unittest.mock import MagicMock
from monocle_test_tools.evals.okahu_eval import OkahuEval


def _make_response(status_code, json_body=None, text=None):
    response = MagicMock(spec=requests.Response)
    response.status_code = status_code
    if json_body is not None:
        response.json.return_value = json_body
        response.text = text if text is not None else "<json>"
    else:
        response.json.side_effect = ValueError("not json")
        response.text = text if text is not None else ""
    return response


class TestRaiseForEvalResponse:
    """Unit tests for OkahuEval._raise_for_eval_response — maps HTTP errors
    on eval-job submission to clean AssertionError messages."""

    def test_400_with_json_error_field_extracts_message(self):
        response = _make_response(
            400,
            json_body={"error": "Template must have a 'name' field (string)"},
        )
        exc = requests.HTTPError(response=response)

        with pytest.raises(AssertionError) as info:
            OkahuEval._raise_for_eval_response(response, exc)

        assert "Custom template validation failed" in str(info.value)
        assert "Template must have a 'name' field (string)" in str(info.value)

    def test_400_with_json_missing_error_field_falls_back_to_text(self):
        response = _make_response(
            400,
            json_body={"detail": "something else"},
            text='{"detail": "something else"}',
        )
        exc = requests.HTTPError(response=response)

        with pytest.raises(AssertionError) as info:
            OkahuEval._raise_for_eval_response(response, exc)

        assert "Custom template validation failed" in str(info.value)
        assert '{"detail": "something else"}' in str(info.value)

    def test_400_with_non_json_body_uses_raw_text(self):
        response = _make_response(400, json_body=None, text="Bad Request")
        exc = requests.HTTPError(response=response)

        with pytest.raises(AssertionError) as info:
            OkahuEval._raise_for_eval_response(response, exc)

        assert "Custom template validation failed: Bad Request" in str(info.value)

    def test_400_with_empty_body_uses_placeholder(self):
        response = _make_response(400, json_body=None, text="")
        exc = requests.HTTPError(response=response)

        with pytest.raises(AssertionError) as info:
            OkahuEval._raise_for_eval_response(response, exc)

        assert "Custom template validation failed: <empty body>" in str(info.value)

    def test_404_uses_trace_not_found_message(self):
        response = _make_response(404, json_body=None, text="not found")
        exc = requests.HTTPError(response=response)

        with pytest.raises(AssertionError) as info:
            OkahuEval._raise_for_eval_response(response, exc)

        assert "Trace not found in evaluation service" in str(info.value)

    def test_other_status_code_uses_generic_message(self):
        response = _make_response(503, json_body=None, text="Service Unavailable")
        exc = requests.HTTPError(response=response)

        with pytest.raises(AssertionError) as info:
            OkahuEval._raise_for_eval_response(response, exc)

        assert "Evaluation service returned HTTP 503" in str(info.value)
        assert "Service Unavailable" in str(info.value)
