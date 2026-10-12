"""Tests for the response evaluator's pure parts (prompt building, tool metadata).

judge() itself needs a live model and is covered by the worked example integration test.
"""
from monocle_test_tools.harness.evaluator import ResponseEvaluator, Verdict


def _evaluator():
    return ResponseEvaluator(model=None, success_criteria="Flight is booked")


def test_verdict_requires_met_and_reason():
    verdict = Verdict(met=True, reason="booking confirmed")
    assert verdict.met is True
    assert verdict.confidence is None


def test_judge_prompt_contains_the_success_criteria():
    prompt = _evaluator().build_judge_prompt("Your flight is booked.", "")
    assert "Flight is booked" in prompt


def test_judge_prompt_contains_the_target_response():
    prompt = _evaluator().build_judge_prompt("Your flight is booked.", "")
    assert "Your flight is booked." in prompt


def test_judge_prompt_contains_the_transcript_when_given():
    prompt = _evaluator().build_judge_prompt("ok", "turn 1 tester> hi")
    assert "turn 1 tester> hi" in prompt


def test_judge_prompt_marks_an_empty_transcript():
    prompt = _evaluator().build_judge_prompt("ok", "")
    assert "(none)" in prompt


def test_judge_prompt_stringifies_a_non_string_response():
    prompt = _evaluator().build_judge_prompt({"status": "booked"}, "")
    assert "booked" in prompt


def test_as_tool_exposes_evaluate_response():
    tool = _evaluator().as_tool()
    assert tool.name == "evaluate_response"
    assert "success criteria" in tool.description
