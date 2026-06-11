"""
Unit tests for the built-in non-LLM (deterministic) evaluators.

These evaluators are pure functions of a span's input/output: no API key, no
network, no model download. The tests cover positive, negative and edge cases for
each evaluator, plus their registration in the eval manager.
"""
import pytest

from monocle_test_tools.evals.regex_match_eval import RegexMatchEval
from monocle_test_tools.evals.json_validity_eval import JSONValidityEval
from monocle_test_tools.evals.keyword_presence_eval import KeywordPresenceEval
from monocle_test_tools.evals.exact_match_eval import ExactMatchEval
from monocle_test_tools.evals.pii_detection_eval import PIIDetectionEval
from monocle_test_tools.evals.readability_eval import ReadabilityEval
from monocle_test_tools.evals.token_overlap_eval import TokenOverlapEval
from monocle_test_tools.evals.bleu_eval import BleuEval
from monocle_test_tools.evals.rouge_eval import RougeEval
from monocle_test_tools.evals.eval_manager import get_evaluator, NON_LLM_EVALS


# ---------------------------------------------------------------------------
# RegexMatchEval
# ---------------------------------------------------------------------------
class TestRegexMatchEval:
    def test_search_match(self):
        ev = RegexMatchEval(eval_options={"pattern": r"\d{4}"})
        result = ev.evaluate({"output": "order 1234 confirmed"})
        assert result == {"match": 1.0, "match_count": 1.0}

    def test_no_match(self):
        ev = RegexMatchEval(eval_options={"pattern": r"\d{4}"})
        result = ev.evaluate({"output": "no digits here"})
        assert result == {"match": 0.0, "match_count": 0.0}

    def test_match_count_multiple(self):
        ev = RegexMatchEval(eval_options={"pattern": r"\d+"})
        result = ev.evaluate({"output": "1 then 2 then 3"})
        assert result["match"] == 1.0
        assert result["match_count"] == 3.0

    def test_ignore_case(self):
        ev = RegexMatchEval(eval_options={"pattern": r"hello", "ignore_case": True})
        assert ev.evaluate({"output": "HELLO world"})["match"] == 1.0

    def test_case_sensitive_by_default(self):
        ev = RegexMatchEval(eval_options={"pattern": r"hello"})
        assert ev.evaluate({"output": "HELLO world"})["match"] == 0.0

    def test_full_match(self):
        ev = RegexMatchEval(eval_options={"pattern": r"\d{4}", "full_match": True})
        assert ev.evaluate({"output": "1234"})["match"] == 1.0
        assert ev.evaluate({"output": "x1234"})["match"] == 0.0

    def test_missing_pattern_raises(self):
        ev = RegexMatchEval(eval_options={})
        with pytest.raises(ValueError):
            ev.evaluate({"output": "anything"})

    def test_missing_output_raises(self):
        ev = RegexMatchEval(eval_options={"pattern": r"\d"})
        with pytest.raises(ValueError):
            ev.evaluate({})


# ---------------------------------------------------------------------------
# JSONValidityEval
# ---------------------------------------------------------------------------
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

    def test_already_parsed_dict(self):
        ev = JSONValidityEval(eval_options={})
        assert ev.evaluate({"output": {"id": 5}})["valid_json"] == 1.0

    def test_missing_output_raises(self):
        ev = JSONValidityEval(eval_options={})
        with pytest.raises(ValueError):
            ev.evaluate({})


# ---------------------------------------------------------------------------
# KeywordPresenceEval
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# ExactMatchEval
# ---------------------------------------------------------------------------
class TestExactMatchEval:
    def test_exact(self):
        ev = ExactMatchEval(eval_options={})
        assert ev.evaluate({"input": "yes", "output": "yes"}) == {"exact_match": 1.0}

    def test_whitespace_and_case_normalized(self):
        ev = ExactMatchEval(eval_options={})
        assert ev.evaluate({"input": "Hello World", "output": "hello   world"}) == {"exact_match": 1.0}

    def test_case_sensitive_when_disabled(self):
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


# ---------------------------------------------------------------------------
# PIIDetectionEval
# ---------------------------------------------------------------------------
class TestPIIDetectionEval:
    def test_detects_email_and_phone(self):
        ev = PIIDetectionEval(eval_options={})
        result = ev.evaluate({"output": "reach me at a@b.com or 555-123-4567"})
        assert result["pii_free"] == 0.0
        assert result["pii_count"] == 2.0
        assert result["pii_breakdown"]["email"] == 1.0
        assert result["pii_breakdown"]["phone"] == 1.0

    def test_clean_text(self):
        ev = PIIDetectionEval(eval_options={})
        result = ev.evaluate({"output": "no personal info here"})
        assert result["pii_free"] == 1.0
        assert result["pii_count"] == 0.0

    def test_ssn(self):
        ev = PIIDetectionEval(eval_options={})
        assert ev.evaluate({"output": "ssn 123-45-6789"})["pii_breakdown"]["ssn"] == 1.0

    def test_pii_types_subset(self):
        ev = PIIDetectionEval(eval_options={"pii_types": ["email"]})
        result = ev.evaluate({"output": "a@b.com and 555-123-4567"})
        # Only email detector runs; phone is ignored.
        assert "phone" not in result["pii_breakdown"]
        assert result["pii_count"] == 1.0

    def test_custom_pattern(self):
        ev = PIIDetectionEval(eval_options={"custom_patterns": {"badge": r"BADGE-\d+"}})
        result = ev.evaluate({"output": "id BADGE-99"})
        assert result["pii_breakdown"]["badge"] == 1.0

    def test_missing_output_raises(self):
        ev = PIIDetectionEval(eval_options={})
        with pytest.raises(ValueError):
            ev.evaluate({})


# ---------------------------------------------------------------------------
# ReadabilityEval
# ---------------------------------------------------------------------------
class TestReadabilityEval:
    def test_simple_text_is_easy(self):
        ev = ReadabilityEval(eval_options={})
        result = ev.evaluate({"output": "The cat sat on the mat. It was a sunny day."})
        assert result["word_count"] == 11.0
        assert result["sentence_count"] == 2.0
        # Short common words score high on reading ease.
        assert result["flesch_reading_ease"] > 80.0

    def test_complex_text_is_harder(self):
        ev = ReadabilityEval(eval_options={})
        simple = ev.evaluate({"output": "The dog ran fast. It was fun."})
        complex_ = ev.evaluate(
            {"output": "Consequently, the multifaceted instrumentation framework "
                       "necessitates comprehensive observability infrastructure."}
        )
        assert complex_["flesch_reading_ease"] < simple["flesch_reading_ease"]

    def test_empty_output(self):
        ev = ReadabilityEval(eval_options={})
        result = ev.evaluate({"output": ""})
        assert result["word_count"] == 0.0
        assert result["flesch_reading_ease"] == 0.0

    def test_missing_output_raises(self):
        ev = ReadabilityEval(eval_options={})
        with pytest.raises(ValueError):
            ev.evaluate({})

    def test_syllable_counter(self):
        assert ReadabilityEval._count_syllables("cat") == 1
        assert ReadabilityEval._count_syllables("hello") == 2
        assert ReadabilityEval._count_syllables("readability") >= 4
        assert ReadabilityEval._count_syllables("make") == 1  # silent trailing e


# ---------------------------------------------------------------------------
# TokenOverlapEval
# ---------------------------------------------------------------------------
class TestTokenOverlapEval:
    def test_partial_overlap(self):
        ev = TokenOverlapEval(eval_options={})
        result = ev.evaluate({"input": "the quick brown fox", "output": "the brown fox jumped"})
        assert result["precision"] == 0.75
        assert result["recall"] == 0.75
        assert result["f1"] == 0.75

    def test_identical(self):
        ev = TokenOverlapEval(eval_options={})
        result = ev.evaluate({"input": "hello world", "output": "hello world"})
        assert result == {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    def test_disjoint(self):
        ev = TokenOverlapEval(eval_options={})
        assert ev.evaluate({"input": "abc def", "output": "xyz qrs"}) == {
            "precision": 0.0, "recall": 0.0, "f1": 0.0
        }

    def test_case_insensitive_default(self):
        ev = TokenOverlapEval(eval_options={})
        assert ev.evaluate({"input": "Hello", "output": "hello"})["f1"] == 1.0

    def test_empty_returns_zero(self):
        ev = TokenOverlapEval(eval_options={})
        assert ev.evaluate({"input": "", "output": "hello"}) == {
            "precision": 0.0, "recall": 0.0, "f1": 0.0
        }

    def test_missing_arg_raises(self):
        ev = TokenOverlapEval(eval_options={})
        with pytest.raises(ValueError):
            ev.evaluate({"output": "hello"})


# ---------------------------------------------------------------------------
# BleuEval
# ---------------------------------------------------------------------------
class TestBleuEval:
    def test_perfect_match_is_one(self):
        ev = BleuEval(eval_options={})
        result = ev.evaluate({"input": "the cat sat on the mat", "output": "the cat sat on the mat"})
        assert result["bleu"] == pytest.approx(1.0, abs=1e-6)
        assert result["brevity_penalty"] == 1.0

    def test_no_overlap_is_zero(self):
        ev = BleuEval(eval_options={})
        result = ev.evaluate({"input": "alpha beta gamma delta", "output": "one two three four"})
        assert result["bleu"] == 0.0

    def test_partial_overlap_between_zero_and_one(self):
        ev = BleuEval(eval_options={})
        result = ev.evaluate({"input": "the cat sat on the mat", "output": "the cat sat on the rug"})
        assert 0.0 < result["bleu"] < 1.0

    def test_brevity_penalty_for_short_candidate(self):
        ev = BleuEval(eval_options={"max_n": 1})
        # All candidate unigrams match, but candidate is much shorter than reference.
        result = ev.evaluate({"input": "the cat sat on the mat today", "output": "the cat"})
        assert result["brevity_penalty"] < 1.0
        assert result["bleu"] < 1.0

    def test_unigram_precision_only(self):
        ev = BleuEval(eval_options={"max_n": 1})
        result = ev.evaluate({"input": "the cat sat", "output": "the cat sat"})
        assert result["precision_1"] == pytest.approx(1.0)
        assert "precision_2" not in result

    def test_empty_returns_zero(self):
        ev = BleuEval(eval_options={})
        assert ev.evaluate({"input": "", "output": "hello"})["bleu"] == 0.0

    def test_case_insensitive_default(self):
        ev = BleuEval(eval_options={})
        result = ev.evaluate({"input": "The Cat", "output": "the cat"})
        assert result["bleu"] == pytest.approx(1.0, abs=1e-6)

    def test_missing_arg_raises(self):
        ev = BleuEval(eval_options={})
        with pytest.raises(ValueError):
            ev.evaluate({"output": "hello"})


# ---------------------------------------------------------------------------
# RougeEval
# ---------------------------------------------------------------------------
class TestRougeEval:
    def test_perfect_match_all_ones(self):
        ev = RougeEval(eval_options={})
        result = ev.evaluate({"input": "the cat sat on the mat", "output": "the cat sat on the mat"})
        for key in ("rouge1_f", "rouge2_f", "rougeL_f"):
            assert result[key] == 1.0

    def test_no_overlap_all_zero(self):
        ev = RougeEval(eval_options={})
        result = ev.evaluate({"input": "alpha beta gamma", "output": "one two three"})
        assert result["rouge1_f"] == 0.0
        assert result["rougeL_f"] == 0.0

    def test_rouge1_precision_recall(self):
        ev = RougeEval(eval_options={"rouge_types": ["rouge1"]})
        # reference 4 tokens, candidate 2 tokens, both overlap.
        result = ev.evaluate({"input": "the cat sat down", "output": "the cat"})
        assert result["rouge1_p"] == 1.0       # both candidate tokens are in reference
        assert result["rouge1_r"] == 0.5       # 2 of 4 reference tokens matched

    def test_rougeL_uses_subsequence_order(self):
        ev = RougeEval(eval_options={"rouge_types": ["rougeL"]})
        # LCS of "a b c d" and "a c b d" is "a b d" or "a c d" -> length 3.
        result = ev.evaluate({"input": "a b c d", "output": "a c b d"})
        assert result["rougeL_f"] == pytest.approx(0.75, abs=1e-4)

    def test_subset_of_types(self):
        ev = RougeEval(eval_options={"rouge_types": ["rouge2"]})
        result = ev.evaluate({"input": "the cat sat", "output": "the cat sat"})
        assert set(result.keys()) == {"rouge2_p", "rouge2_r", "rouge2_f"}

    def test_invalid_type_raises(self):
        ev = RougeEval(eval_options={"rouge_types": ["rougeX"]})
        with pytest.raises(ValueError):
            ev.evaluate({"input": "a b", "output": "a b"})

    def test_case_insensitive_default(self):
        ev = RougeEval(eval_options={"rouge_types": ["rouge1"]})
        assert ev.evaluate({"input": "The Cat", "output": "the cat"})["rouge1_f"] == 1.0

    def test_missing_arg_raises(self):
        ev = RougeEval(eval_options={})
        with pytest.raises(ValueError):
            ev.evaluate({"input": "hello"})


# ---------------------------------------------------------------------------
# eval_manager registration
# ---------------------------------------------------------------------------
class TestEvalManagerRegistration:
    @pytest.mark.parametrize("key,cls", [
        ("regex_match", RegexMatchEval),
        ("json_validity", JSONValidityEval),
        ("keyword_presence", KeywordPresenceEval),
        ("exact_match", ExactMatchEval),
        ("pii_detection", PIIDetectionEval),
        ("readability", ReadabilityEval),
        ("token_overlap", TokenOverlapEval),
        ("bleu", BleuEval),
        ("rouge", RougeEval),
    ])
    def test_get_evaluator_resolves_string_key(self, key, cls):
        ev = get_evaluator(key, {})
        assert isinstance(ev, cls)

    def test_all_keys_registered(self):
        expected = {
            "regex_match", "json_validity", "keyword_presence", "exact_match",
            "pii_detection", "readability", "token_overlap", "bleu", "rouge",
        }
        assert expected == set(NON_LLM_EVALS.keys())

    def test_eval_options_passed_through(self):
        ev = get_evaluator("regex_match", {"pattern": r"\d+"})
        assert ev.pattern == r"\d+"

    def test_passthrough_existing_instance(self):
        instance = RegexMatchEval(eval_options={"pattern": r"x"})
        assert get_evaluator(instance, {}) is instance

    def test_invalid_key_raises(self):
        with pytest.raises(ValueError):
            get_evaluator("does_not_exist", {})
