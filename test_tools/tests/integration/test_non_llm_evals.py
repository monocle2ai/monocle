"""
Integration tests for the core built-in non-LLM (deterministic) evaluators.

These run a real ADK travel agent end-to-end, collect the real OpenTelemetry
spans Monocle produces, extract each span's input/output with the same
``trace_utils`` helpers the framework uses, and then exercise the core
non-LLM evaluators against that genuine trace data.

The ``_run_evaluation`` helper mirrors ``MonocleValidator._evaluate_span``: it
builds ``eval_args`` from the ``Evaluation.args`` selectors, runs the evaluator,
and applies the configured comparer — so these tests validate the full
``Evaluation`` -> evaluator -> comparer pipeline, not just the evaluators in
isolation (that is covered by ``tests/unit/test_non_llm_evals.py``).

Like the other tests under ``tests/integration``, these require ADK to be
installed and valid Google GenAI credentials to be configured; they run in CI.
"""
from asyncio import sleep
import pytest

import monocle_test_tools.trace_utils as trace_utils
from monocle_test_tools.schema import Evaluation, EvalInputs
from test_common.adk_travel_agent import root_agent

BOOKING_PROMPT = "Book a flight from San Jose to Seattle for 27th Nov 2025."


def _get_turn_span(asserter):
    """Return the final ``agentic.turn`` span collected during the agent run."""
    turn_spans = [
        span for span in asserter.validator.spans
        if span.attributes.get("span.type") == "agentic.turn"
    ]
    if not turn_spans:
        pytest.skip("No agentic.turn span was produced by the agent run.")
    return turn_spans[-1]


def _run_evaluation(span, evaluation: Evaluation):
    """Mirror MonocleValidator._evaluate_span: build eval_args, evaluate, compare.

    Returns a ``(passed, actual_result)`` tuple.
    """
    eval_args = {}
    for arg in evaluation.args:
        if arg == EvalInputs.INPUT:
            eval_args["input"] = trace_utils.get_input_from_span(span)
        elif arg == EvalInputs.OUTPUT:
            eval_args["output"] = trace_utils.get_output_from_span(span)
        elif arg == EvalInputs.AGENT_DESCRIPTION:
            eval_args["agent_description"] = trace_utils.get_agent_description_from_span(span)
    actual = evaluation.eval.evaluate(eval_args)
    passed = evaluation.comparer.compare(evaluation.expected_result, actual)
    return passed, actual


# ---------------------------------------------------------------------------
# One evaluator per test, each against the real agentic.turn span.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_turn_output_has_booking_keywords(monocle_trace_asserter):
    """The confirmation should mention the booking and contain no error language."""
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk", BOOKING_PROMPT)
    span = _get_turn_span(monocle_trace_asserter)

    evaluation = Evaluation(
        eval="keyword_presence",
        eval_options={
            "required_keywords": ["flight"],
            "forbidden_keywords": ["password", "traceback"],
        },
        args=[EvalInputs.OUTPUT],
        expected_result={"required_coverage": 1.0, "forbidden_absent": 1.0},
        comparer="metric",
    )
    passed, actual = _run_evaluation(span, evaluation)
    assert passed, f"Keyword presence check failed: {actual}"


@pytest.mark.asyncio
async def test_turn_output_matches_destination_regex(monocle_trace_asserter):
    """The confirmation should reference the requested destination city."""
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk", BOOKING_PROMPT)
    span = _get_turn_span(monocle_trace_asserter)

    evaluation = Evaluation(
        eval="regex_match",
        eval_options={"pattern": r"seattle|san\s+jose", "ignore_case": True},
        args=[EvalInputs.OUTPUT],
        expected_result={"match": 1.0},
        comparer="metric",
    )
    passed, actual = _run_evaluation(span, evaluation)
    assert passed, f"Expected destination city not found in output: {actual}"


@pytest.mark.asyncio
async def test_turn_output_is_not_verbatim_request(monocle_trace_asserter):
    """The agent should respond, not echo the request verbatim (exact_match == 0)."""
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk", BOOKING_PROMPT)
    span = _get_turn_span(monocle_trace_asserter)

    evaluator = Evaluation(
        eval="exact_match",
        args=[EvalInputs.INPUT, EvalInputs.OUTPUT],
        expected_result={"exact_match": 0.0},
        comparer="metric",
    )
    _, actual = _run_evaluation(span, evaluator)
    assert actual == {"exact_match": 0.0}, f"Output unexpectedly equals the request: {actual}"


@pytest.mark.asyncio
async def test_turn_output_is_prose_not_json(monocle_trace_asserter):
    """The user-facing turn output is natural-language prose, not raw JSON."""
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk", BOOKING_PROMPT)
    span = _get_turn_span(monocle_trace_asserter)

    evaluator = Evaluation(
        eval="json_validity",
        args=[EvalInputs.OUTPUT],
        expected_result={"valid_json": 0.0},
        comparer="metric",
    )
    _, actual = _run_evaluation(span, evaluator)
    assert actual["valid_json"] == 0.0, f"Turn output was unexpectedly raw JSON: {actual}"


# ---------------------------------------------------------------------------
# All core evaluators against a single agent run (cost-efficient smoke test).
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_all_non_llm_evals_on_single_run(monocle_trace_asserter):
    """Exercise every core non-LLM evaluator + comparer against one real turn span."""
    await monocle_trace_asserter.run_agent_async(root_agent, "google_adk", BOOKING_PROMPT)
    span = _get_turn_span(monocle_trace_asserter)

    evaluations = [
        Evaluation(eval="keyword_presence",
                   eval_options={"required_keywords": ["flight"], "forbidden_keywords": ["traceback"]},
                   args=[EvalInputs.OUTPUT],
                   expected_result={"required_coverage": 1.0, "forbidden_absent": 1.0}, comparer="metric"),
        Evaluation(eval="regex_match",
                   eval_options={"pattern": r"seattle|san\s+jose", "ignore_case": True},
                   args=[EvalInputs.OUTPUT], expected_result={"match": 1.0}, comparer="metric"),
        Evaluation(eval="exact_match", args=[EvalInputs.INPUT, EvalInputs.OUTPUT],
                   expected_result={"exact_match": 0.0}, comparer="metric"),
        Evaluation(eval="json_validity", args=[EvalInputs.OUTPUT],
                   expected_result={"valid_json": 0.0}, comparer="metric"),
    ]

    failures = []
    for evaluation in evaluations:
        passed, actual = _run_evaluation(span, evaluation)
        if not passed:
            failures.append((evaluation.eval.__class__.__name__, evaluation.expected_result, actual))
    assert not failures, f"Non-LLM eval failures: {failures}"
    await sleep(2)  # be gentle with rate limits, consistent with other integration tests


if __name__ == "__main__":
    pytest.main([__file__])