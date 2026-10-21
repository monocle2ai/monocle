"""Copy-ready example: drive monocle eval test cases from a CSV.

Real usage — a teammate writes this stub once; non-engineers edit the CSV:

    import os
    from monocle_test_tools import monocle_csv_cases

    TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "hallucination_test.json")

    @monocle_csv_cases("cases.example.csv")
    def test_cases(monocle_trace_asserter, case):
        case.run(monocle_trace_asserter.with_evaluation("okahu"),
                 template_path=TEMPLATE_PATH)

The smoke test below runs offline: it only checks the CSV loads and maps.
"""
import os

from monocle_test_tools import load_cases_from_csv

_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cases.example.csv")


def test_example_csv_loads():
    cases = load_cases_from_csv(_CSV)
    assert len(cases) >= 2
    assert cases[0].case_id
    # every eval row carries a label (guaranteed by the loader)
    assert cases[0].expected is not None or cases[0].not_expected is not None
