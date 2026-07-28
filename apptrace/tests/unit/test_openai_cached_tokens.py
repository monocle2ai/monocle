"""Unit tests for cached token extraction in update_span_from_llm_response.

OpenAI exposes cached token counts in two places depending on the API used:
  - Chat Completions API: usage.prompt_tokens_details.cached_tokens
  - Responses API:        usage.input_tokens_details.cached_tokens

Both should be captured as ``cache_read_input_tokens`` in the metadata dict.
"""
import unittest
from types import SimpleNamespace

from monocle_apptrace.instrumentation.metamodel.openai._helper import (
    update_span_from_llm_response,
)


def _make_chat_response(
    prompt_tokens=100,
    completion_tokens=50,
    total_tokens=150,
    cached_tokens=None,
):
    """Simulate a ChatCompletion response object (attribute-style usage)."""
    prompt_details = SimpleNamespace(cached_tokens=cached_tokens) if cached_tokens is not None else SimpleNamespace()
    usage = SimpleNamespace(
        completion_tokens=completion_tokens,
        prompt_tokens=prompt_tokens,
        total_tokens=total_tokens,
        prompt_tokens_details=prompt_details,
        input_tokens_details=None,
    )
    return SimpleNamespace(usage=usage)


def _make_responses_api_response(
    input_tokens=100,
    output_tokens=50,
    total_tokens=150,
    cached_tokens=None,
):
    """Simulate a Responses API response object (uses input_tokens_details)."""
    input_details = SimpleNamespace(cached_tokens=cached_tokens) if cached_tokens is not None else SimpleNamespace()
    usage = SimpleNamespace(
        output_tokens=output_tokens,
        input_tokens=input_tokens,
        total_tokens=total_tokens,
        prompt_tokens_details=None,
        input_tokens_details=input_details,
    )
    return SimpleNamespace(usage=usage)


def _make_dict_response(prompt_tokens=100, completion_tokens=50, total_tokens=150,
                        cached_tokens=None, use_input_details=False):
    """Simulate a dict-form response (serialized JSON payload)."""
    details_key = "input_tokens_details" if use_input_details else "prompt_tokens_details"
    usage: dict = {
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "total_tokens": total_tokens,
    }
    if cached_tokens is not None:
        usage[details_key] = {"cached_tokens": cached_tokens}
    return {"usage": usage}


class TestBasicTokenExtraction(unittest.TestCase):
    def test_chat_completion_basic(self):
        response = _make_chat_response(prompt_tokens=80, completion_tokens=40, total_tokens=120)
        result = update_span_from_llm_response(response)
        self.assertEqual(result["prompt_tokens"], 80)
        self.assertEqual(result["completion_tokens"], 40)
        self.assertEqual(result["total_tokens"], 120)
        self.assertNotIn("cache_read_input_tokens", result)

    def test_no_response(self):
        result = update_span_from_llm_response(None)
        self.assertEqual(result, {})


class TestCachedTokensAttributeStyle(unittest.TestCase):
    """Chat Completions API – attribute-style usage object."""

    def test_cached_tokens_via_prompt_tokens_details(self):
        response = _make_chat_response(prompt_tokens=200, completion_tokens=60, total_tokens=260, cached_tokens=80)
        result = update_span_from_llm_response(response)
        self.assertEqual(result["cache_read_input_tokens"], 80)
        self.assertEqual(result["prompt_tokens"], 200)
        self.assertEqual(result["completion_tokens"], 60)
        self.assertEqual(result["total_tokens"], 260)

    def test_zero_cached_tokens_not_emitted(self):
        """cached_tokens=0 is a valid value and should still be captured."""
        response = _make_chat_response(prompt_tokens=100, completion_tokens=50, total_tokens=150, cached_tokens=0)
        result = update_span_from_llm_response(response)
        # 0 is falsy but it's a real value; it IS returned since we check `is not None`
        self.assertEqual(result.get("cache_read_input_tokens"), 0)

    def test_no_cached_tokens_field(self):
        """prompt_tokens_details present but no cached_tokens attribute."""
        usage = SimpleNamespace(
            completion_tokens=50,
            prompt_tokens=100,
            total_tokens=150,
            prompt_tokens_details=SimpleNamespace(),  # no cached_tokens attr
            input_tokens_details=None,
        )
        response = SimpleNamespace(usage=usage)
        result = update_span_from_llm_response(response)
        self.assertNotIn("cache_read_input_tokens", result)

    def test_no_prompt_tokens_details(self):
        """No prompt_tokens_details and no input_tokens_details."""
        usage = SimpleNamespace(
            completion_tokens=50,
            prompt_tokens=100,
            total_tokens=150,
            prompt_tokens_details=None,
            input_tokens_details=None,
        )
        response = SimpleNamespace(usage=usage)
        result = update_span_from_llm_response(response)
        self.assertNotIn("cache_read_input_tokens", result)


class TestCachedTokensResponsesAPI(unittest.TestCase):
    """Responses API – uses input_tokens_details instead of prompt_tokens_details."""

    def test_cached_tokens_via_input_tokens_details(self):
        response = _make_responses_api_response(input_tokens=300, output_tokens=70, total_tokens=370, cached_tokens=120)
        result = update_span_from_llm_response(response)
        self.assertEqual(result["cache_read_input_tokens"], 120)

    def test_prompt_tokens_details_takes_priority(self):
        """If both detail objects exist, prompt_tokens_details wins."""
        prompt_details = SimpleNamespace(cached_tokens=50)
        input_details = SimpleNamespace(cached_tokens=99)
        usage = SimpleNamespace(
            completion_tokens=50,
            prompt_tokens=100,
            total_tokens=150,
            prompt_tokens_details=prompt_details,
            input_tokens_details=input_details,
        )
        response = SimpleNamespace(usage=usage)
        result = update_span_from_llm_response(response)
        self.assertEqual(result["cache_read_input_tokens"], 50)


class TestCachedTokensDictStyle(unittest.TestCase):
    """Dict-form responses (e.g. parsed JSON from serialized text)."""

    def test_cached_tokens_via_prompt_tokens_details_dict(self):
        response = _make_dict_response(prompt_tokens=200, completion_tokens=60, total_tokens=260, cached_tokens=80)
        result = update_span_from_llm_response(response)
        self.assertEqual(result["cache_read_input_tokens"], 80)
        self.assertEqual(result["prompt_tokens"], 200)
        self.assertEqual(result["completion_tokens"], 60)

    def test_cached_tokens_via_input_tokens_details_dict(self):
        response = _make_dict_response(cached_tokens=55, use_input_details=True)
        result = update_span_from_llm_response(response)
        self.assertEqual(result["cache_read_input_tokens"], 55)

    def test_no_details_dict(self):
        response = _make_dict_response()
        result = update_span_from_llm_response(response)
        self.assertNotIn("cache_read_input_tokens", result)


if __name__ == "__main__":
    unittest.main()
