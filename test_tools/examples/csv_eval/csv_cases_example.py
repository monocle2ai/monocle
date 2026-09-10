"""Copy-ready example — Layer 0: drive monocle eval cases from a CSV, values in the file.

The simplest form: the fact-set CSV and the judge are written right here. One row =
one eval test.

Running it — the key must reach the environment. This file accepts it three ways, in
precedence order:

    # 1) one-shot — the key prefixes the command on one line, no separate step:
    OKAHU_API_KEY=sk-... pytest test_tools/examples/csv_eval/csv_cases_example.py

    # 2) export once, then run as many times as you like:
    export OKAHU_API_KEY=sk-...
    pytest test_tools/examples/csv_eval/csv_cases_example.py

    # 3) put it in monocle's own env file — .env.monocle (current dir) or
    #    ~/.monocle/.env — which this file loads automatically (block below):
    echo 'OKAHU_API_KEY=sk-...' >> .env.monocle
    pytest test_tools/examples/csv_eval/csv_cases_example.py

A plain .env is NOT auto-loaded (python-dotenv isn't a dependency); source it into
your shell yourself first: set -a; source your.env; set +a

Each row's `fact_id` is a real Okahu trace id and `expected` is a label the judge can
emit. See cases.example.csv for the column layout.

The judge here is a **custom template** — a JSON file committed next to this example
(hallucination_test.json). A built-in Okahu template is shown as a commented-out
alternative in the test below. See test_tools/README.md ("CSV eval test cases") for the
template schema.

Evolution path (still pure monocle, no extra tooling):
  * to supply the CSV path and template from the ENVIRONMENT instead of hardcoding
    them, see csv_cases_env_example.py;
  * to supply them as pytest command-line FLAGS, see csv_cases_parametrized_example.py.
"""
import os

import pytest
from monocle_test_tools import monocle_csv_cases

# If OKAHU_API_KEY isn't already exported/inlined, fall back to monocle's own env files
# (.env.monocle in the current dir, then ~/.monocle/.env) via monocle_apptrace — already
# a dependency, so no extra install. An inline/exported key wins (setdefault won't clobber).
from monocle_apptrace.instrumentation.common.utils import get_monocle_env_value
_okahu_key = get_monocle_env_value("OKAHU_API_KEY")
if _okahu_key:
    os.environ.setdefault("OKAHU_API_KEY", _okahu_key)

# These examples replay against a live Okahu tenant; skip cleanly with no key set.
pytestmark = pytest.mark.skipif(
    not os.getenv("OKAHU_API_KEY"),
    reason="requires OKAHU_API_KEY (these examples replay against a live Okahu tenant)",
)

# Custom template, committed next to this file. Resolved from __file__ (not the CWD) so
# it's found no matter where pytest is invoked from.
TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "hallucination_test.json")


@monocle_csv_cases("cases.example.csv")  # path resolves next to this file
def test_cases(monocle_trace_asserter, case):
    # The stub owns everything constant across the sheet: the evaluator and the judge.
    # Here the judge is a CUSTOM template — your own LLM-as-a-judge JSON, graded via
    # template_path. The row's `expected` labels must be values this template can emit.
    case.run(
        monocle_trace_asserter.with_evaluation("okahu"),
        template_path=TEMPLATE_PATH,
    )

    # Built-in template (alternative) — grade with a template registered in Okahu by
    # name, so no template file is needed. To use it instead, replace the call above with:
    #
    #     case.run(
    #         monocle_trace_asserter.with_evaluation("okahu"),
    #         eval_name="hallucination",
    #     )


if __name__ == "__main__":
    pytest.main([__file__])
