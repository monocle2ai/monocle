"""Unit tests for capturing response_format in Monocle's data.input span event for LiteLLM.

response_format is a valid top-level arg to litellm.completion(), but by the time
the instrumented backend is called, litellm has moved it into optional_params.
So Monocle must read kwargs["optional_params"]["response_format"], not the top-level kwarg.
"""
import json
import unittest

from pydantic import BaseModel

from monocle_apptrace.instrumentation.metamodel.litellm._helper import (
    extract_messages,
    extract_response_format,
)
from monocle_apptrace.instrumentation.metamodel.litellm.entities.inference import (
    INFERENCE,
)


class _Sentiment(BaseModel):
    label: str
    explanation: str


def _make_arguments(response_format):
    """Shape arguments exactly as task_wrapper hands them to the provider method."""
    optional_params = {"extra_body": {}}
    if response_format is not None:
        optional_params["response_format"] = response_format
    return {
        "kwargs": {
            "messages": [{"role": "user", "content": "What is coffee?"}],
            "optional_params": optional_params,
        }
    }


def _run_data_input(arguments):
    """Run every data.input accessor from the live INFERENCE metamodel."""
    out = {}
    for event in INFERENCE["events"]:
        if event["name"] == "data.input":
            for attr in event["attributes"]:
                out[attr["attribute"]] = attr["accessor"](arguments)
    return out


class TestLiteLLMResponseFormatHelper(unittest.TestCase):
    """Direct coverage of extract_response_format serialization."""

    def test_dict_response_format(self):
        result = extract_response_format(
            {"optional_params": {"response_format": {"type": "json_object"}}}
        )
        self.assertEqual(json.loads(result), {"type": "json_object"})

    def test_pydantic_response_format(self):
        result = extract_response_format(
            {"optional_params": {"response_format": _Sentiment}}
        )
        schema = json.loads(result)
        # Serialized as the model's JSON schema.
        self.assertIn("label", schema["properties"])
        self.assertIn("explanation", schema["properties"])

    def test_absent_response_format_returns_none(self):
        self.assertIsNone(
            extract_response_format({"optional_params": {"extra_body": {}}})
        )

    def test_missing_optional_params_returns_none(self):
        self.assertIsNone(extract_response_format({}))

    def test_none_optional_params_returns_none(self):
        self.assertIsNone(extract_response_format({"optional_params": None}))

    def test_top_level_kwarg_is_not_used(self):
        # Guard against a regression that reads a top-level kwarg instead of the
        # nested optional_params location (where litellm actually puts it).
        self.assertIsNone(
            extract_response_format({"response_format": {"type": "json_object"}})
        )

    def test_bedrock_json_tool_call_response_format(self):
        # Bedrock Converse converts a Pydantic response_format into a json_tool_call
        # tool placed in optional_params["tools"] (with json_mode=true).
        kwargs = {
            "optional_params": {
                "json_mode": True,
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "json_tool_call",
                            "parameters": {"properties": {"label": {"type": "string"}}},
                        },
                    }
                ],
            }
        }
        schema = json.loads(extract_response_format(kwargs))
        self.assertIn("label", schema["json_schema"]["schema"]["properties"])

    def test_anthropic_json_tool_call_response_format(self):
        # Anthropic converts a Pydantic response_format into a json_tool_call tool in
        # optional_params["tools"], but with its own shape: {"name", "input_schema"}.
        kwargs = {
            "optional_params": {
                "json_mode": True,
                "tools": [
                    {
                        "name": "json_tool_call",
                        "input_schema": {"properties": {"label": {"type": "string"}}},
                    }
                ],
            }
        }
        schema = json.loads(extract_response_format(kwargs))
        self.assertIn("label", schema["json_schema"]["schema"]["properties"])


class TestLiteLLMInferenceDataInput(unittest.TestCase):
    """Coverage through the live INFERENCE output processor."""

    def test_response_format_attribute_present_dict(self):
        result = _run_data_input(_make_arguments({"type": "json_object"}))
        self.assertIn("response_format", result)
        self.assertEqual(json.loads(result["response_format"]), {"type": "json_object"})
        # The existing input attribute must still be populated.
        self.assertEqual(result["input"], ['{"user": "What is coffee?"}'])

    def test_response_format_attribute_present_pydantic(self):
        result = _run_data_input(_make_arguments(_Sentiment))
        schema = json.loads(result["response_format"])
        self.assertIn("label", schema["properties"])

    def test_response_format_attribute_none_when_absent(self):
        result = _run_data_input(_make_arguments(None))
        self.assertIn("response_format", result)
        self.assertIsNone(result["response_format"])


class TestLiteLLMExtractMessages(unittest.TestCase):
    """Input extraction must work for both the OpenAI and Azure LiteLLM call shapes.

    OpenAI's (a)completion is invoked with messages=..., but Azure's async acompletion
    receives the already-transformed request as data={"messages": [...]}. Both must yield
    the same captured input so prod eval inference spans capture the input for Azure too.
    """

    def test_extract_messages_from_top_level_messages(self):
        kwargs = {"messages": [{"role": "user", "content": "What is coffee?"}]}
        self.assertEqual(extract_messages(kwargs), ['{"user": "What is coffee?"}'])

    def test_extract_messages_from_azure_data_dict(self):
        # Azure async acompletion shape: messages live under data["messages"].
        kwargs = {"data": {"messages": [{"role": "user", "content": "What is coffee?"}]}}
        self.assertEqual(extract_messages(kwargs), ['{"user": "What is coffee?"}'])

    def test_extract_messages_top_level_takes_precedence(self):
        kwargs = {
            "messages": [{"role": "user", "content": "top level"}],
            "data": {"messages": [{"role": "user", "content": "nested"}]},
        }
        self.assertEqual(extract_messages(kwargs), ['{"user": "top level"}'])

    def test_extract_messages_empty_when_no_messages(self):
        self.assertEqual(extract_messages({"data": {}}), [])
        self.assertEqual(extract_messages({}), [])


if __name__ == "__main__":
    unittest.main()
