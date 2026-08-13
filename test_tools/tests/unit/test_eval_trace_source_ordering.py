"""Trace-source wiring must not depend on fluent-call ORDER.

Regression tests for the "first test of a session loses its scenario" bug.

`with_evaluation()` used to snapshot `validator._trace_source` into the evaluator at
construction time. The CSV adapter (and any test written as
`case.run(asserter.with_evaluation("okahu"), ...)`) configures the evaluation BEFORE
the trace source is known, so on the first test of a session the evaluator was built
with an empty trace source. That produced two wrong behaviours against a real
service:

  * `shadow_eval` was sent as True, so the service ran a shadow eval and returned
    `{"message": "Job submitted", "result": []}` -> the client raised
    "Unexpected response format ... Expected 'result' key".
  * `_trace_exported` was False, so the client tried to re-export a trace that the
    okahu source already holds.

Later tests in the same session accidentally worked because `MonocleValidator` is a
process-wide singleton: the first test's `with_trace_source` left the source set.
"""
import json
from unittest.mock import MagicMock, patch

import pytest

from monocle_test_tools.evals.okahu_eval import OkahuEval
from monocle_test_tools.fluent_api import TraceAssertion
from monocle_test_tools.validator import MonocleValidator

TEMPLATE = {
    "name": "t",
    "eval_prompt": "x",
    "structure_output": {
        "label": {"enums": ["a", "b"], "description": "x"},
        "explanation": {"description": "x"},
    },
}
JUDGE = {"label": "a", "explanation": "why", "total_tokens": 7}


def _resp():
    m = MagicMock()
    m.headers = {"Content-Type": "application/json"}
    m.raise_for_status.return_value = None
    m.json.return_value = {"job_id": "interactive_x_1",
                           "result": [{"result": json.dumps(JUDGE)}]}
    return m


def _fake_import_traces(self, trace_source=None, **kwargs):
    """Stand-in for MonocleValidator.import_traces: records the source the same way
    the real implementation does, without touching the network."""
    self._trace_source = trace_source


@pytest.fixture(autouse=True)
def _reset_shared_state():
    # TraceAssertion carries class-level state; MonocleValidator is a singleton whose
    # _trace_source starts empty in a real session. Reset both so each test here
    # genuinely reproduces "first test of the session".
    TraceAssertion._assertion_errors = []
    TraceAssertion._eval_report = None
    TraceAssertion._okahu_filter = None
    MonocleValidator()._trace_source = ""
    yield
    TraceAssertion._assertion_errors = []
    TraceAssertion._eval_report = None
    TraceAssertion._okahu_filter = None
    MonocleValidator()._trace_source = ""


def _eval_with_order(monkeypatch, evaluation_first: bool):
    """Run one check_eval through the real OkahuEval, returning (asserter, post_mock,
    export_mock). `evaluation_first` selects the fluent call order."""
    span = MagicMock()
    span.attributes = {"workflow.name": "wf"}
    span.start_time = 1_000_000_000
    span.end_time = 2_000_000_000
    monkeypatch.setenv("OKAHU_API_KEY", "k")

    asserter = TraceAssertion(filtered_spans=[span])

    with patch.object(MonocleValidator, "import_traces", _fake_import_traces):
        if evaluation_first:
            asserter = asserter.with_evaluation("okahu")
            asserter = asserter.with_trace_source("okahu", id="traceid", workflow_name="wf")
        else:
            asserter = asserter.with_trace_source("okahu", id="traceid", workflow_name="wf")
            asserter = asserter.with_evaluation("okahu")

    with patch.object(OkahuEval, "export_trace", return_value="traceid") as export_mock, \
         patch.object(OkahuEval, "enumerate_fact_ids", return_value=["traceid"]), \
         patch("monocle_test_tools.evals.okahu_eval.OkahuEvalResultExporter"), \
         patch("monocle_test_tools.evals.okahu_eval.requests.post",
               return_value=_resp()) as post_mock:
        asserter.check_eval(template=TEMPLATE, expected="a")

    return asserter, post_mock, export_mock


def test_shadow_eval_is_false_when_evaluation_configured_before_trace_source(monkeypatch):
    asserter, post_mock, _ = _eval_with_order(monkeypatch, evaluation_first=True)

    assert not asserter.has_assertions(), asserter.get_assertion_messages()
    params = post_mock.call_args.kwargs["params"]
    assert params["shadow_eval"] is False, (
        "an okahu trace source must never be evaluated as a shadow eval; "
        "the service returns an empty result for shadow evals"
    )


def test_okahu_source_is_not_re_exported_when_evaluation_configured_first(monkeypatch):
    _, _, export_mock = _eval_with_order(monkeypatch, evaluation_first=True)

    export_mock.assert_not_called()


def test_shadow_eval_is_false_when_trace_source_configured_first(monkeypatch):
    # The already-working order must keep working.
    asserter, post_mock, _ = _eval_with_order(monkeypatch, evaluation_first=False)

    assert not asserter.has_assertions(), asserter.get_assertion_messages()
    assert post_mock.call_args.kwargs["params"]["shadow_eval"] is False


def test_non_okahu_source_still_shadow_evals(monkeypatch):
    # Guard against over-correcting: a file/local source IS a shadow eval.
    span = MagicMock()
    span.attributes = {"workflow.name": "wf"}
    span.start_time = 1_000_000_000
    span.end_time = 2_000_000_000
    monkeypatch.setenv("OKAHU_API_KEY", "k")

    asserter = TraceAssertion(filtered_spans=[span])
    with patch.object(MonocleValidator, "import_traces", _fake_import_traces):
        asserter = asserter.with_evaluation("okahu")
        asserter = asserter.with_trace_source("file", path="somewhere.json")

    with patch.object(OkahuEval, "export_trace", return_value="traceid"), \
         patch.object(OkahuEval, "enumerate_fact_ids", return_value=["traceid"]), \
         patch("monocle_test_tools.evals.okahu_eval.OkahuEvalResultExporter"), \
         patch("monocle_test_tools.evals.okahu_eval.requests.post",
               return_value=_resp()) as post_mock:
        asserter.check_eval(template=TEMPLATE, expected="a")

    assert post_mock.call_args.kwargs["params"]["shadow_eval"] is True
