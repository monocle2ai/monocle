"""Live end-to-end filtered eval + custom template smoke test (opt-in).

Skipped unless OKAHU_API_KEY (+ OKAHU_*_ENDPOINT) and RUN_LIVE_FILTERED=1 are set.
Submits ONE async filtered job (no trace_id) scoped by workflow + time window, lets
the server discover every matching fact, and asserts a blanket expectation over all
of them with a custom template. This run is also where the job-detail `results`
fact-id field name is confirmed against a live job (plan Phase 5 Step 2).

Run against stage:

    RUN_LIVE_FILTERED=1 MONOCLE_EVAL_MATRIX=1 OKAHU_API_KEY=... \
      OKAHU_EVALUATION_ENDPOINT=https://evals-stage.okahu.co/api \
      OKAHU_API_ENDPOINT=https://api-stage.okahu.co \
      SMOKE_WORKFLOW=... SMOKE_START=... SMOKE_END=... \
      pytest tests/integration/test_filtered_custom_eval_live.py -v -s
"""
import os

import pytest

from monocle_test_tools.pytest_plugin import monocle_trace_asserter  # noqa: F401 (fixture)

pytestmark = pytest.mark.skipif(
    not os.getenv("OKAHU_API_KEY") or not os.getenv("RUN_LIVE_FILTERED"),
    reason="requires OKAHU_API_KEY + OKAHU_*_ENDPOINT and RUN_LIVE_FILTERED=1")

TEMPLATE = {  # or load a committed hallucination_test.json
    "name": "mtt_filtered_smoke",
    "eval_prompt": "Return label no_hallucination for any response.",
    "structure_output": {
        "label": {"description": "one of the labels",
                  "enums": ["no_hallucination", "minor_hallucination", "major_hallucination"]},
        "explanation": {"description": "why"},
    },
}


def test_filtered_custom_template_smoke(monocle_trace_asserter):
    # Blanket gate over every fact the job discovers (no id list).
    (monocle_trace_asserter
        .with_filtered_source("okahu", workflow_name=os.environ["SMOKE_WORKFLOW"],
                              start_time=os.environ["SMOKE_START"], end_time=os.environ["SMOKE_END"])
        .with_evaluation("okahu")
        .check_eval_filtered(template=TEMPLATE, expected="no_hallucination", min_facts=1))
    report = monocle_trace_asserter.get_filtered_eval_report()
    assert report["summary"]["total"] >= 1
