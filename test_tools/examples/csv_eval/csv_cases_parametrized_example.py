"""Copy-ready example — Layer 1b: CSV path and template from pytest FLAGS.

Same capability as csv_cases_env_example.py, but the values come from command-line
flags (--fact-set / --template) — the discoverable form for a human at a terminal.
Flags are NOT readable at import, so this uses pytest_generate_tests (not the
@monocle_csv_cases decorator) and loads the CSV with load_cases_from_csv.

Needs the sibling conftest.py (it registers --fact-set / --template). No committed
defaults: supply both flags or the test skips.

Two different mechanisms here: OKAHU_API_KEY is an ENVIRONMENT variable (inline,
exported, or from monocle's own .env.monocle / ~/.monocle/.env, which this file loads
automatically — block below), while the CSV and template are pytest command-line FLAGS.
So the key goes in the environment; the inputs are --fact-set / --template on the command:

    # one-shot — key inline, inputs as flags:
    OKAHU_API_KEY=sk-... pytest test_tools/examples/csv_eval/csv_cases_parametrized_example.py \
        --fact-set=test_tools/examples/csv_eval/cases.example.csv \
        --template=test_tools/examples/csv_eval/hallucination_test.json

    # or export the key once, then just pass the flags:
    export OKAHU_API_KEY=sk-...
    pytest test_tools/examples/csv_eval/csv_cases_parametrized_example.py \
        --fact-set=test_tools/examples/csv_eval/cases.example.csv \
        --template=test_tools/examples/csv_eval/hallucination_test.json

    # --template may be a custom .json path (as above) OR a built-in name like "hallucination".
"""
import os

import pytest
from monocle_test_tools import load_cases_from_csv

# If OKAHU_API_KEY isn't already exported/inlined, fall back to monocle's own env files
# (.env.monocle in the current dir, then ~/.monocle/.env) via monocle_apptrace — already
# a dependency, so no extra install. An inline/exported key wins (setdefault won't clobber).
from monocle_apptrace.instrumentation.common.utils import get_monocle_env_value
_okahu_key = get_monocle_env_value("OKAHU_API_KEY")
if _okahu_key:
    os.environ.setdefault("OKAHU_API_KEY", _okahu_key)

pytestmark = pytest.mark.skipif(
    not os.getenv("OKAHU_API_KEY"),
    reason="requires OKAHU_API_KEY (these examples replay against a live Okahu tenant)",
)


def is_custom_template(value):
    """Custom template = a .json file (which must exist). Anything else = a built-in name."""
    if value.endswith(".json") or os.sep in value:
        if not os.path.isfile(value):
            raise pytest.UsageError(
                f"--template looks like a custom template file but none exists: {value}")
        return True
    return False


def pytest_generate_tests(metafunc):
    if "case" not in metafunc.fixturenames:
        return
    fact_set = metafunc.config.getoption("--fact-set")
    template = metafunc.config.getoption("--template")
    if not (fact_set and template):
        # No committed defaults: emit one clearly-skipped item instead of a hard error.
        metafunc.parametrize(
            "case",
            [pytest.param(None, marks=pytest.mark.skip(
                reason="pass --fact-set=<csv> and --template=<path|name> to run"))],
        )
        return
    cases = load_cases_from_csv(fact_set)
    metafunc.parametrize("case", cases, ids=[c.case_id for c in cases])


def test_cases(monocle_trace_asserter, case, request):
    # Both flags are present here — pytest_generate_tests skipped the run otherwise.
    template = request.config.getoption("--template")
    asserter = monocle_trace_asserter.with_evaluation("okahu")
    if is_custom_template(template):
        case.run(asserter, template_path=template)   # custom template file
    else:
        case.run(asserter, eval_name=template)       # built-in template name


if __name__ == "__main__":
    pytest.main([__file__])
