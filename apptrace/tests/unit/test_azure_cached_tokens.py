"""Unit tests for cached token extraction in the azure-ai-inference metamodel.

Azure OpenAI deployments served through the azure-ai-inference endpoint report
cached prompt tokens under ``usage.prompt_tokens_details.cached_tokens`` (or
``input_tokens_details`` on some deployments). Because the typed
``CompletionsUsage`` model may not surface these, the value can also arrive in
the model's ``additional_properties`` bag or in a dict-form usage payload.

All shapes should be captured as ``cache_read_input_tokens`` in the metadata,
matching the OpenAI metamodel and the token-summary tooling.

(Azure OpenAI accessed via ``openai.AzureOpenAI`` uses the OpenAI metamodel and
is covered by ``test_openai_cached_tokens.py``.)
"""
import unittest
from types import SimpleNamespace

from monocle_apptrace.instrumentation.metamodel.azureaiinference._helper import (
    update_span_from_llm_response,
)


def _usage(prompt=100, completion=50, total=150, **extra):
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        **extra,
    )


class TestAzureBasicTokenExtraction(unittest.TestCase):
    def test_basic_no_cache(self):
        result = SimpleNamespace(usage=_usage())
        attrs = update_span_from_llm_response(result)
        self.assertEqual(attrs["prompt_tokens"], 100)
        self.assertEqual(attrs["completion_tokens"], 50)
        self.assertEqual(attrs["total_tokens"], 150)
        self.assertNotIn("cache_read_input_tokens", attrs)

    def test_no_usage(self):
        result = SimpleNamespace()
        attrs = update_span_from_llm_response(result)
        self.assertNotIn("cache_read_input_tokens", attrs)


class TestAzureCachedTokens(unittest.TestCase):
    def test_prompt_tokens_details_attr(self):
        usage = _usage(prompt_tokens_details=SimpleNamespace(cached_tokens=64))
        attrs = update_span_from_llm_response(SimpleNamespace(usage=usage))
        self.assertEqual(attrs["cache_read_input_tokens"], 64)

    def test_input_tokens_details_attr(self):
        usage = _usage(input_tokens_details=SimpleNamespace(cached_tokens=77))
        attrs = update_span_from_llm_response(SimpleNamespace(usage=usage))
        self.assertEqual(attrs["cache_read_input_tokens"], 77)

    def test_zero_cached_tokens_captured(self):
        usage = _usage(prompt_tokens_details=SimpleNamespace(cached_tokens=0))
        attrs = update_span_from_llm_response(SimpleNamespace(usage=usage))
        self.assertEqual(attrs.get("cache_read_input_tokens"), 0)

    def test_additional_properties_bag(self):
        """Typed model drops the field; it lands in additional_properties."""
        usage = _usage(
            additional_properties={"prompt_tokens_details": {"cached_tokens": 88}}
        )
        attrs = update_span_from_llm_response(SimpleNamespace(usage=usage))
        self.assertEqual(attrs["cache_read_input_tokens"], 88)

    def test_dict_form_details(self):
        usage = _usage(prompt_tokens_details={"cached_tokens": 42})
        attrs = update_span_from_llm_response(SimpleNamespace(usage=usage))
        self.assertEqual(attrs["cache_read_input_tokens"], 42)

    def test_no_cached_tokens_field(self):
        usage = _usage(prompt_tokens_details=SimpleNamespace())
        attrs = update_span_from_llm_response(SimpleNamespace(usage=usage))
        self.assertNotIn("cache_read_input_tokens", attrs)


class _MappingModel(dict):
    """Mimics azure-core Model: declared fields are attributes, unmodeled fields
    are only reachable via mapping access (``obj[name]`` / ``obj.get(name)``)."""

    def __init__(self, attrs, mapping):
        super().__init__(mapping)
        for k, v in attrs.items():
            setattr(self, k, v)


class TestAzureCoreModelStyle(unittest.TestCase):
    """The real azure-ai-inference CompletionsUsage does not declare
    prompt_tokens_details as an attribute; it is only reachable via mapping."""

    def test_cached_tokens_via_mapping_access(self):
        usage = _MappingModel(
            attrs={"prompt_tokens": 300, "completion_tokens": 40, "total_tokens": 340},
            mapping={"prompt_tokens_details": {"cached_tokens": 256}},
        )
        attrs = update_span_from_llm_response(SimpleNamespace(usage=usage))
        self.assertEqual(attrs["cache_read_input_tokens"], 256)
        self.assertEqual(attrs["prompt_tokens"], 300)

    def test_input_tokens_details_via_mapping_access(self):
        usage = _MappingModel(
            attrs={"prompt_tokens": 300, "completion_tokens": 40, "total_tokens": 340},
            mapping={"input_tokens_details": {"cached_tokens": 111}},
        )
        attrs = update_span_from_llm_response(SimpleNamespace(usage=usage))
        self.assertEqual(attrs["cache_read_input_tokens"], 111)


if __name__ == "__main__":
    unittest.main()
