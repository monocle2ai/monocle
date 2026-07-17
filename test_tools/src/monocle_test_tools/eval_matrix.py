"""Opt-in eval-result-matrix recorder for pytest.

This module provides:
  - `build_eval_matrix_row`: a pure function that turns the per-test
    `_last_eval` stash (see `TraceAssertion.check_eval`) into a single
    matrix row with a fixed schema.
  - pytest plugin hooks (`pytest_addoption`, `pytest_sessionfinish`) and a
    `maybe_record_eval_row` helper that `pytest_plugin.py` wires into the
    `monocle_trace_asserter` fixture teardown.

The feature is OFF by default: unless `--monocle-eval-matrix` is passed or
`MONOCLE_EVAL_MATRIX` is set, no rows are recorded and no file is written.
"""
import json
import os
from datetime import datetime, timezone
from typing import Any, Optional

# Module-level accumulator for recorded rows. Only ever appended to when the
# feature is enabled (see `is_enabled` / `maybe_record_eval_row`).
_records: list[dict[str, Any]] = []

DEFAULT_OUTPUT_PATH = "test-eval-replay-matrix.json"


def build_eval_matrix_row(run_id: str, scenario: str, last_eval: dict, passed: bool) -> dict:
    """Build a single eval-result-matrix row from a `_last_eval` stash.

    `last_eval` is expected to look like the dict stashed by
    `TraceAssertion.check_eval`:
        {"trace_id": str, "expected": ..., "fact_name": str, "label": ...,
         "explanation": ..., "judge_output": dict, "total_tokens": ...}

    Returns a dict with exactly these keys (schema is consumed downstream,
    do not rename): run_id, scenario, trace_id, expected, actual, status,
    explanation, total_tokens, claim_verdicts, hallucination_types,
    entity_match_check.
    """
    last_eval = last_eval or {}
    judge_output = last_eval.get("judge_output") or {}

    actual = last_eval.get("label") or ""
    if passed:
        status = "pass"
    elif actual:
        status = "drift"
    else:
        status = "error"

    return {
        "run_id": run_id,
        "scenario": scenario,
        "trace_id": last_eval.get("trace_id"),
        "expected": last_eval.get("expected"),
        "actual": actual,
        "status": status,
        "explanation": last_eval.get("explanation"),
        "total_tokens": last_eval.get("total_tokens"),
        "claim_verdicts": judge_output.get("claim_verdicts") or [],
        "hallucination_types": judge_output.get("hallucination_types") or [],
        "entity_match_check": judge_output.get("entity_match_check") or "",
    }


_FALSY_ENV_VALUES = ("0", "false", "no", "off", "")
_BOOLEAN_ENV_VALUES = ("1", "true", "yes", "on")


def _is_env_truthy(value: Optional[str]) -> bool:
    """Whether `MONOCLE_EVAL_MATRIX` should be treated as "set" (enabling the
    recorder), as opposed to unset or explicitly disabled (e.g. "0"/"false").
    """
    if value is None:
        return False
    return value.strip().lower() not in _FALSY_ENV_VALUES


def resolve_output_path(option_value: Optional[str], env_value: Optional[str]) -> Optional[str]:
    """Resolve whether the matrix recorder is enabled and, if so, the output path.

    Priority: `--monocle-eval-matrix` option value, else `MONOCLE_EVAL_MATRIX`
    if it looks like a path (i.e. isn't just a boolean-ish "1"/"true"/...),
    else the default filename. Returns the output path (str) if enabled,
    otherwise None.
    """
    if option_value:
        return option_value
    if _is_env_truthy(env_value):
        if env_value.strip().lower() in _BOOLEAN_ENV_VALUES:
            return DEFAULT_OUTPUT_PATH
        return env_value
    return None


def is_enabled(option_value: Optional[str], env_value: Optional[str]) -> bool:
    return resolve_output_path(option_value, env_value) is not None


def reset_records() -> None:
    """Clear the module-level accumulator. Exposed for test isolation."""
    _records.clear()


def get_records() -> list[dict[str, Any]]:
    return _records


def record_row(row: dict[str, Any]) -> None:
    _records.append(row)


def record_eval_row_for(config, request, trace_assertion) -> None:
    """Record a matrix row for `trace_assertion`'s last eval, if enabled.

    Self-skips when the feature is disabled or when the asserter has no
    `_last_eval` (i.e. the test never called `check_eval`).
    """
    option_value = config.getoption("monocle_eval_matrix", default=None)
    env_value = os.environ.get("MONOCLE_EVAL_MATRIX")
    if not is_enabled(option_value, env_value):
        return

    last_eval = getattr(trace_assertion, "_last_eval", None)
    if last_eval is None:
        return

    node = request.node
    callspec = getattr(node, "callspec", None)
    scenario = callspec.id if callspec is not None else node.name

    rep_call = getattr(node, "rep_call", None)
    passed = bool(rep_call and rep_call.passed)

    run_id = os.getenv("RUN_ID") or os.getenv("LOCAL_RUN_ID") or "local"

    row = build_eval_matrix_row(run_id=run_id, scenario=scenario, last_eval=last_eval, passed=passed)
    record_row(row)


def pytest_addoption(parser) -> None:
    parser.addoption(
        "--monocle-eval-matrix",
        dest="monocle_eval_matrix",
        nargs="?",
        const=DEFAULT_OUTPUT_PATH,
        default=None,
        help=(
            "Opt-in: record an eval-result matrix (trace_id, expected/actual "
            "label, tokens, judge output) for every test that calls "
            "check_eval, and write it to the given path (default: "
            f"{DEFAULT_OUTPUT_PATH}) at session end. Also enabled by setting "
            "the MONOCLE_EVAL_MATRIX env var. Off by default."
        ),
    )


def pytest_sessionfinish(session) -> None:
    config = session.config
    option_value = config.getoption("monocle_eval_matrix", default=None)
    env_value = os.environ.get("MONOCLE_EVAL_MATRIX")
    output_path = resolve_output_path(option_value, env_value)
    if not output_path:
        return

    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "records": get_records(),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
