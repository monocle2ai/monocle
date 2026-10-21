"""Integration: drive a CSV of test cases through the fluent asserter against okahu.

Requires a real Okahu environment. Update the placeholder constants below with
real values (same convention as test_okahu_trace_assertions.py) before running:

    pytest tests/integration/test_csv_cases_okahu.py -m integration

The CSV adapter fixes source=okahu and replays by trace id; the evaluator and
template are supplied by the stub (here the built-in "hallucination" template
via eval_name), exactly as with the fluent check_eval path.
"""
import textwrap

import pytest

from monocle_test_tools import load_cases_from_csv

pytestmark = pytest.mark.integration

# Update these with real values from your Okahu environment to run the test.
DEMO_WORKFLOW_NAME = "Okahu-Loader-Demo"
PLACEHOLDER_TRACE_ID = "642dbd9d0dfcfdbdc8849f67f34c8a19"
EXPECTED_LABEL = "major_hallucination"


@pytest.fixture()
def cases_csv(tmp_path):
    path = tmp_path / "cases.csv"
    path.write_text(
        textwrap.dedent(f"""\
            case_id,fact_id,workflow_name,expected
            cc_t01,{PLACEHOLDER_TRACE_ID},{DEMO_WORKFLOW_NAME},{EXPECTED_LABEL}
        """),
        encoding="utf-8",
    )
    return str(path)


def test_csv_case_replays_and_evaluates(monocle_trace_asserter, cases_csv):
    cases = load_cases_from_csv(cases_csv)
    assert len(cases) == 1

    # Stub owns evaluator + template selection; CSV owns fact_id/workflow/expected.
    cases[0].run(
        monocle_trace_asserter.with_evaluation("okahu"),
        eval_name="hallucination",
    )

    # Failures (if any) are collected on the asserter and surfaced by the
    # pytest plugin's makereport hook; a passing eval leaves no assertions.
    assert not monocle_trace_asserter.has_assertions()


if __name__ == "__main__":
    pytest.main([__file__])
