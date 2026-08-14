"""Live, opt-in integration test for eval discovery.

Importing this package runs ``tests/integration/__init__.py``, which pins the
stage Okahu endpoints + key (via ``os.environ.setdefault``), so this exercises
the real ``/v1/eval/templates`` + ``/evals/query`` path on stage.
"""
import os

import pytest

pytestmark = pytest.mark.skipif(
    not (os.getenv("OKAHU_API_KEY") and os.getenv("OKAHU_EVALUATION_ENDPOINT")
         and os.getenv("OKAHU_API_ENDPOINT")),
    reason="Okahu credentials not configured",
)

# Known sample fact captured in docs/eval_prompt_data.md. Stage retention may have
# purged it; the test skips gracefully if its trace can no longer be loaded.
SAMPLE_TRACE_ID = "c9fb4edad4860b57bfc9ef64e1c3d451"
SAMPLE_WORKFLOW = "karmehr-okahu-monocle"


def test_discovery_runs_end_to_end_against_okahu():
    """Discovery must run against live Okahu and stay non-fatal.

    Either it pins discovered evals (baseline comment + check_eval line) or it
    records a note (no evals / skipped) — never a crash. To assert real pinned
    evals, point SAMPLE_TRACE_ID/SAMPLE_WORKFLOW at a current fact with stored evals.
    """
    from monocle_test_tools.test_generator import TestGenerator
    try:
        gen = TestGenerator.from_okahu(trace_id=SAMPLE_TRACE_ID, workflow_name=SAMPLE_WORKFLOW)
    except Exception as exc:
        pytest.skip(f"sample trace not loadable from Okahu (likely purged): {exc}")

    code = gen.generate_test_code()

    assert (
        "# discovered from fact" in code
        or "# No existing evals found on this fact" in code
        or "# eval discovery skipped:" in code
    ), "eval discovery did not run / left no outcome in the generated code"

    # When evals were discovered, each carries a check_eval line + baseline comment.
    if "# discovered from fact" in code:
        assert "check_eval(" in code


def test_discovery_query_path_from_local_trace():
    """Exercise the live templates + /evals/query HTTP path with a reliable local
    trace as input (span loading can't fail on stale cloud data). The local
    trace's fact id isn't on stage, so discovery finds nothing or is skipped —
    the point is that the real query path runs and stays non-fatal."""
    from pathlib import Path
    from monocle_test_tools.test_generator import TestGenerator

    trace_path = "tests/unit/traces/trace1.json"
    if not Path(trace_path).exists():
        pytest.skip(f"Trace file {trace_path} not found")

    gen = TestGenerator.from_json_file(trace_path)
    code = gen.generate_test_code()

    assert (
        "# discovered from fact" in code
        or "# No existing evals found on this fact" in code
        or "# eval discovery skipped:" in code
    ), "eval discovery did not run against the local trace"
