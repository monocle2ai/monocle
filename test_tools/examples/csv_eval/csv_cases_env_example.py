"""Copy-ready example — Layer 1a: CSV path and template from the ENVIRONMENT.

Same as csv_cases_example.py, but the fact-set CSV and the template come from
environment variables instead of being hardcoded — so one file serves many fact
sets / templates without editing. This is the form CI naturally uses (the workflow
injects the env). Env is readable at import, so the @decorator stays.

No committed defaults: FACT_SET and TEMPLATE must both be set or the test skips.

All three values (the key + the two inputs) are read from the process environment.
Inline them on one command, or export them first:

    # one-shot — the three vars prefix the command as one logical line:
    OKAHU_API_KEY=sk-... \
    FACT_SET=test_tools/examples/csv_eval/cases.example.csv \
    TEMPLATE=test_tools/examples/csv_eval/hallucination_test.json \
        pytest test_tools/examples/csv_eval/csv_cases_env_example.py

    # or export once, then run (handy across repeated runs / a whole CI job):
    export OKAHU_API_KEY=sk-...
    export FACT_SET=test_tools/examples/csv_eval/cases.example.csv
    export TEMPLATE=test_tools/examples/csv_eval/hallucination_test.json
    pytest test_tools/examples/csv_eval/csv_cases_env_example.py

    # TEMPLATE may be a custom .json path (as above) OR a built-in name like "hallucination".
    # FACT_SET is resolved against your current directory (absolutised below).

The KEY additionally loads from monocle's own env files — .env.monocle (current dir) or
~/.monocle/.env — automatically (block below), so you can keep it out of the command and
just supply FACT_SET / TEMPLATE. FACT_SET and TEMPLATE are NOT read from those files. A
plain .env is not auto-loaded either.
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

pytestmark = pytest.mark.skipif(
    not os.getenv("OKAHU_API_KEY"),
    reason="requires OKAHU_API_KEY (these examples replay against a live Okahu tenant)",
)

# Both values come from the environment — no committed defaults; blank counts as unset.
FACT_SET = os.getenv("FACT_SET", "")
TEMPLATE = os.getenv("TEMPLATE", "")


def is_custom_template(value):
    """Custom template = a .json file (which must exist). Anything else = a built-in name."""
    if value.endswith(".json") or os.sep in value:
        if not os.path.isfile(value):
            raise pytest.UsageError(
                f"TEMPLATE looks like a custom template file but none exists: {value}")
        return True
    return False


if FACT_SET and TEMPLATE:
    # Absolutise so FACT_SET is read relative to your CWD (what an operator expects),
    # not relative to this file's directory (monocle_csv_cases' default for a bare path).
    @monocle_csv_cases(os.path.abspath(FACT_SET))
    def test_cases(monocle_trace_asserter, case):
        asserter = monocle_trace_asserter.with_evaluation("okahu")
        if is_custom_template(TEMPLATE):
            case.run(asserter, template_path=TEMPLATE)   # custom template file
        else:
            case.run(asserter, eval_name=TEMPLATE)       # built-in template name
else:
    @pytest.mark.skip(reason="set FACT_SET and TEMPLATE (custom .json path or built-in name) to run")
    def test_cases():
        pass


if __name__ == "__main__":
    pytest.main([__file__])
