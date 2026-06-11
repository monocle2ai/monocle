"""
Unit tests for the core built-in non-LLM (deterministic) evaluators.

These tests cover the four core evaluator classes Monocle currently exposes:
regex_match, json_validity, keyword_presence, and exact_match.
"""

import pytest

from monocle_test_tools.evals.eval_manager import NON_LLM_EVALS, get_evaluator
from monocle_test_tools.evals.exact_match_eval import ExactMatchEval
from monocle_test_tools.evals.json_validity_eval import JSONValidityEval
from monocle_test_tools.evals.keyword_presence_eval import KeywordPresenceEval
from monocle_test_tools.evals.regex_match_eval import RegexMatchEval


class TestRegexMatchEval:
    def test_search_match(self):
        ev = RegexMatchEval(eval_options={"pattern": r"\d{4}"})
        result = ev.evaluate({"output": "order 1234 confirmed"})
        assert result == {"match": 1.0, "match_count": 1.0}

    def test_no_match(self):
        ev = RegexMatchEval(eval_options={"pattern": r"\d{4}"})
        result = ev.evaluate({"output": "no digits here"})
        assert result == {"match": 0.0, "match_count": 0.0}

    def test_multiple_matches(self):
        ev = RegexMatchEval(eval_options={"pattern": r"\d+"})
        result = ev.evaluate({"output": "1 then 2 then 3"})
        assert result["match"] == 1.0
        assert result["match_count"] == 3.0

    def test_full_match(self):
        ev = RegexMatchEval(eval_options={"pattern": r"\d{4}", "full_match": True})
        assert ev.evaluate({"output": "1234"})["match"] == 1.0
        assert ev.evaluate({"output": "x1234"})["match"] == 0.0

    def test_ignore_case(self):
        ev = RegexMatchEval(eval_options={"pattern": r"hello", "ignore_case": True})
        assert ev.evaluate({"output": "HELLO world"})["match"] == 1.0

    def test_missing_pattern_raises(self):
        ev = RegexMatchEval(eval_options={})
        with pytest.raises(ValueError):
            ev.evaluate({"output": "anything"})

    def test_missing_output_raises(self):
        ev = RegexMatchEval(eval_options={"pattern": r"\d"})
        with pytest.raises(ValueError):
            ev.evaluate({})


class TestJSONValidityEval:
    def test_valid_json_no_schema(self):
        ev = JSONValidityEval(eval_options={})
        assert ev.evaluate({"output": '{"id": 5}'}) == {"valid_json": 1.0, "schema_valid": 1.0}

    def test_invalid_json(self):
        ev = JSONValidityEval(eval_options={})
        assert ev.evaluate({"output": "not json"}) == {"valid_json": 0.0, "schema_valid": 0.0}

    def test_schema_pass(self):
        schema = {"type": "object", "required": ["id"]}
        ev = JSONValidityEval(eval_options={"json_schema": schema})
        assert ev.evaluate({"output": '{"id": 5}'}) == {"valid_json": 1.0, "schema_valid": 1.0}

    def test_schema_fail(self):
        schema = {"type": "object", "required": ["id"]}
        ev = JSONValidityEval(eval_options={"json_schema": schema})
        assert ev.evaluate({"output": '{"name": 5}'}) == {"valid_json": 1.0, "schema_valid": 0.0}

    def test_dict_input(self):
        ev = JSONValidityEval(eval_options={})
        assert ev.evaluate({"output": {"id": 5}})["valid_json"] == 1.0

    def test_missing_output_raises(self):
        ev = JSONValidityEval(eval_options={})
        with pytest.raises(ValueError):
            ev.evaluate({})


class TestKeywordPresenceEval:
    def test_all_required_present(self):
        ev = KeywordPresenceEval(eval_options={"required_keywords": ["refund", "processed"]})
        result = ev.evaluate({"output": "your refund is processed"})
        assert result["required_coverage"] == 1.0
        assert result["missing_required"] == 0.0

    def test_partial_required(self):
        ev = KeywordPresenceEval(eval_options={"required_keywords": ["refund", "tomorrow"]})
        result = ev.evaluate({"output": "your refund is processed"})
        assert result["required_coverage"] == 0.5
        assert result["missing_required"] == 1.0

    def test_forbidden_present(self):
        ev = KeywordPresenceEval(eval_options={"forbidden_keywords": ["password"]})
        result = ev.evaluate({"output": "your password is 1234"})
        assert result["forbidden_absent"] == 0.0
        assert result["forbidden_found"] == 1.0

    def test_forbidden_absent(self):
        ev = KeywordPresenceEval(eval_options={"forbidden_keywords": ["password"]})
        result = ev.evaluate({"output": "all good here"})
        assert result["forbidden_absent"] == 1.0

    def test_case_insensitive_default(self):
        ev = KeywordPresenceEval(eval_options={"required_keywords": ["REFUND"]})
        assert ev.evaluate({"output": "refund done"})["required_coverage"] == 1.0

    def test_case_sensitive(self):
        ev = KeywordPresenceEval(eval_options={"required_keywords": ["REFUND"], "case_sensitive": True})
        assert ev.evaluate({"output": "refund done"})["required_coverage"] == 0.0

    def test_no_keywords_defaults_to_pass(self):
        ev = KeywordPresenceEval(eval_options={})
        result = ev.evaluate({"output": "anything"})
        assert result["required_coverage"] == 1.0
        assert result["forbidden_absent"] == 1.0


class TestExactMatchEval:
    def test_exact(self):
        ev = ExactMatchEval(eval_options={})
        assert ev.evaluate({"input": "yes", "output": "yes"}) == {"exact_match": 1.0}

    def test_whitespace_and_case_normalized(self):
        ev = ExactMatchEval(eval_options={})
        assert ev.evaluate({"input": "Hello World", "output": "hello   world"}) == {"exact_match": 1.0}

    def test_ignore_case_can_be_disabled(self):
        ev = ExactMatchEval(eval_options={"ignore_case": False})
        assert ev.evaluate({"input": "Yes", "output": "yes"}) == {"exact_match": 0.0}

    def test_ignore_punctuation(self):
        ev = ExactMatchEval(eval_options={"ignore_punctuation": True})
        assert ev.evaluate({"input": "yes!", "output": "yes"}) == {"exact_match": 1.0}

    def test_mismatch(self):
        ev = ExactMatchEval(eval_options={})
        assert ev.evaluate({"input": "yes", "output": "no"}) == {"exact_match": 0.0}

    def test_missing_arg_raises(self):
        ev = ExactMatchEval(eval_options={})
        with pytest.raises(ValueError):
            ev.evaluate({"output": "yes"})


class TestEvalManagerRegistration:
    @pytest.mark.parametrize(
        "key,cls",
        [
            ("regex_match", RegexMatchEval),
            ("json_validity", JSONValidityEval),
            ("keyword_presence", KeywordPresenceEval),
            ("exact_match", ExactMatchEval),
        ],
    )
    def test_get_evaluator_resolves_string_key(self, key, cls):
        ev = get_evaluator(key, {})
        assert isinstance(ev, cls)

    def test_all_keys_registered(self):
        assert set(NON_LLM_EVALS.keys()) == {
            "regex_match",
            "json_validity",
            "keyword_presence",
            "exact_match",
        }

    def test_eval_options_passed_through(self):
        ev = get_evaluator("regex_match", {"pattern": r"\d+"})
        assert ev.pattern == r"\d+"

    def test_passthrough_existing_instance(self):
        instance = RegexMatchEval(eval_options={"pattern": r"x"})
        assert get_evaluator(instance, {}) is instance

    def test_invalid_key_raises(self):
        with pytest.raises(ValueError):
            get_evaluator("does_not_exist", {})
