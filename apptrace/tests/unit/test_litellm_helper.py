"""Unit tests for capturing response_format in Monocle's data.input span event for LiteLLM.

response_format is a valid top-level arg to litellm.completion(), but by the time
the instrumented backend is called, litellm has moved it into optional_params.
So Monocle must read kwargs["optional_params"]["response_format"], not the top-level kwarg.
"""
import json
import unittest
from types import SimpleNamespace

from pydantic import BaseModel

from monocle_apptrace.instrumentation.metamodel.litellm._helper import (
    extract_finish_reason,
    extract_messages,
    extract_response_format,
    extract_temperature,
    extract_tool_name,
    extract_tool_type,
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


def _run_tool_attrs(arguments):
    """Run the tool.name/tool.type accessors from the live INFERENCE metamodel."""
    out = {}
    for group in INFERENCE["attributes"]:
        for attr in group:
            if attr.get("_comment", "").startswith("Tool"):
                out[attr["attribute"]] = attr["accessor"](arguments)
    return out


def _make_react_response(content, finish_reason="stop"):
    message = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(finish_reason=finish_reason, message=message)
    return SimpleNamespace(choices=[choice])


def _make_native_tool_call_response(tool_name):
    tool_call = SimpleNamespace(function=SimpleNamespace(name=tool_name))
    message = SimpleNamespace(content=None, tool_calls=[tool_call])
    choice = SimpleNamespace(finish_reason="tool_calls", message=message)
    return SimpleNamespace(choices=[choice])


def _run_metadata(arguments):
    """Run the named metadata accessors from the live INFERENCE metamodel."""
    out = {}
    for event in INFERENCE["events"]:
        if event["name"] == "metadata":
            for attr in event["attributes"]:
                # The usage accessor has no "attribute" key; skip it here.
                if "attribute" in attr:
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


class TestLiteLLMExtractTemperature(unittest.TestCase):
    """temperature is moved into optional_params by the time the backend is called."""

    def test_temperature_from_optional_params(self):
        kwargs = {"optional_params": {"temperature": 0.7, "extra_body": {}}}
        self.assertEqual(extract_temperature(kwargs), 0.7)

    def test_temperature_zero_is_captured(self):
        # temperature=0 (deterministic judge) is falsy but must NOT be treated as absent.
        kwargs = {"optional_params": {"temperature": 0}}
        self.assertEqual(extract_temperature(kwargs), 0)

    def test_temperature_from_azure_data_dict(self):
        # Azure async passes an already-built request under data.
        kwargs = {"data": {"temperature": 0.2}}
        self.assertEqual(extract_temperature(kwargs), 0.2)

    def test_optional_params_takes_precedence_over_data(self):
        kwargs = {"optional_params": {"temperature": 0}, "data": {"temperature": 0.9}}
        self.assertEqual(extract_temperature(kwargs), 0)

    def test_top_level_kwarg_fallback(self):
        self.assertEqual(extract_temperature({"temperature": 0.5}), 0.5)

    def test_absent_temperature_returns_none(self):
        self.assertIsNone(extract_temperature({"optional_params": {"extra_body": {}}}))
        self.assertIsNone(extract_temperature({}))

    def test_none_optional_params_returns_none(self):
        self.assertIsNone(extract_temperature({"optional_params": None}))


class TestLiteLLMInferenceMetadata(unittest.TestCase):
    """Coverage through the live INFERENCE metadata processor."""

    def test_temperature_attribute_present(self):
        arguments = {
            "kwargs": {"optional_params": {"temperature": 0}},
            "result": None,
            "exception": None,
        }
        result = _run_metadata(arguments)
        self.assertIn("temperature", result)
        self.assertEqual(result["temperature"], 0)

    def test_temperature_attribute_none_when_absent(self):
        arguments = {
            "kwargs": {"optional_params": {"extra_body": {}}},
            "result": None,
            "exception": None,
        }
        result = _run_metadata(arguments)
        self.assertIn("temperature", result)
        self.assertIsNone(result["temperature"])


class TestLiteLLMReactStyleToolName(unittest.TestCase):
    """CrewAI-style ReAct text tool calls ('Action: <tool>') carry no native
    tool_calls entry. extract_finish_reason already reclassifies these as
    finish_type=tool_call; extract_tool_name/extract_tool_type must agree on
    the tool identity instead of reporting None for a span already typed as
    a tool call.
    """

    def test_react_tool_call_populates_name_and_type(self):
        response = _make_react_response(
            "Thought: I should look this up\nAction: search_tool\nAction Input: {}"
        )
        arguments = {"result": response, "exception": None}
        self.assertEqual(extract_finish_reason(arguments), "tool_calls")
        self.assertEqual(extract_tool_name(arguments), "search_tool")
        self.assertIsNotNone(extract_tool_type(arguments))

    def test_react_tool_call_strips_surrounding_whitespace(self):
        response = _make_react_response("Thought: ok\nAction:   spaced_tool  \n")
        arguments = {"result": response, "exception": None}
        self.assertEqual(extract_tool_name(arguments), "spaced_tool")

    def test_react_final_answer_is_not_a_tool_call(self):
        response = _make_react_response("Thought: done\nAction: Final Answer: 42")
        arguments = {"result": response, "exception": None}
        self.assertEqual(extract_finish_reason(arguments), "stop")
        self.assertIsNone(extract_tool_name(arguments))
        self.assertIsNone(extract_tool_type(arguments))

    def test_plain_stop_without_action_line_returns_none(self):
        response = _make_react_response("just a normal reply")
        arguments = {"result": response, "exception": None}
        self.assertEqual(extract_finish_reason(arguments), "stop")
        self.assertIsNone(extract_tool_name(arguments))

    def test_native_tool_call_still_resolves(self):
        # Regression guard: the ReAct fallback must not shadow the existing
        # native tool_calls path (OpenAI/Bedrock/Azure function calling).
        response = _make_native_tool_call_response("native_tool")
        arguments = {"result": response, "exception": None}
        self.assertEqual(extract_tool_name(arguments), "native_tool")
        self.assertIsNotNone(extract_tool_type(arguments))

    def test_react_tool_call_through_live_inference_metamodel(self):
        response = _make_react_response("Thought: ok\nAction: weather_tool")
        arguments = {"result": response, "exception": None}
        result = _run_tool_attrs(arguments)
        self.assertEqual(result["name"], "weather_tool")
        self.assertIsNotNone(result["type"])


if __name__ == "__main__":
    unittest.main()
