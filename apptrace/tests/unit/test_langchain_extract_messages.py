import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from monocle_apptrace.instrumentation.common.constants import (
    INFERENCE_AGENT_DELEGATION,
    INFERENCE_TOOL_CALL,
    INFERENCE_TURN_END,
)
from monocle_apptrace.instrumentation.metamodel.langchain._helper import (
    agent_inference_type,
    extract_assistant_message,
    extract_messages,
    extract_tool_name,
)


class _Msg:
    def __init__(self, content, type="human"):
        self.content = content
        self.type = type


class _FakeChatModel:
    """Stands in for a Runnable chat model (e.g. ChatOpenAI) that ends up as args[0]."""

    def invoke(self, *a, **kw):
        return None

    async def ainvoke(self, *a, **kw):
        return None


class TestExtractMessagesBoundCall(unittest.TestCase):
    """Normal bound-method call: args are exactly the call's own positional arguments."""

    def test_messages_list_extracted(self):
        args = ([_Msg("system prompt", type="system"), _Msg("hi", type="human")],)
        out = extract_messages(args)
        self.assertEqual(len(out), 2)
        self.assertEqual(json.loads(out[1]), {"human": "hi"})


class TestExtractMessagesSelfInArgs(unittest.TestCase):
    """Some callers (e.g. retry decorators that monkeypatch a class attribute and forward the
    captured call as a plain function invocation) end up passing the model instance itself as
    the leading positional arg, with no `instance` available via normal descriptor binding.
    """

    def test_skips_leading_model_instance_by_identity(self):
        model = _FakeChatModel()
        messages = [_Msg("system prompt", type="system"), _Msg("hi", type="human")]
        args = (model, messages)
        out = extract_messages(args, instance=model)
        self.assertEqual(len(out), 2)
        self.assertEqual(json.loads(out[1]), {"human": "hi"})

    def test_skips_leading_model_instance_by_duck_typing(self):
        # instance=None mirrors what actually happens: Monocle's own instance detection
        # also comes back empty for this call shape, so identity comparison alone can't help.
        model = _FakeChatModel()
        messages = [_Msg("system prompt", type="system"), _Msg("hi", type="human")]
        args = (model, messages, {"configurable": {"thread_id": "t1"}})
        out = extract_messages(args, instance=None)
        self.assertEqual(len(out), 2)
        self.assertEqual(json.loads(out[1]), {"human": "hi"})

    def test_single_arg_model_only_returns_empty_not_crash(self):
        model = _FakeChatModel()
        out = extract_messages((model,), instance=None)
        self.assertEqual(out, [])


class TestExtractMessagesInputShapes(unittest.TestCase):
    """LanguageModelInput also arrives as a raw string or OpenAI-style dict messages
    ([{"role", "content"}]); both must populate data.input, not be dropped."""

    def test_string_input(self):
        # llm.invoke("...") / with_structured_output(...).ainvoke("...")
        self.assertEqual(extract_messages(("say ok in one word",)), ["say ok in one word"])

    def test_string_input_not_iterated_per_character(self):
        # Regression: a str must not fall through to per-character iteration (empty result).
        self.assertNotEqual(extract_messages(("hello",)), [])

    def test_dict_messages(self):
        out = extract_messages(([{"role": "system", "content": "sys"},
                                 {"role": "user", "content": "hi"}],))
        self.assertIn('{"system": "sys"}', out)
        self.assertIn('{"user": "hi"}', out)

    def test_dict_messages_not_dropped(self):
        self.assertNotEqual(extract_messages(([{"role": "user", "content": "hi"}],)), [])

    def test_prompt_value_text(self):
        self.assertEqual(extract_messages((SimpleNamespace(text="hello prompt"),)), ["hello prompt"])

    def test_empty_args(self):
        self.assertEqual(extract_messages(()), [])


FLIGHT = {"name": "book_flight", "args": {"to": "SEA"}, "id": "c1", "type": "tool_call"}
HOTEL = {"name": "book_hotel", "args": {"hotel": "Hyatt"}, "id": "c2", "type": "tool_call"}


def _arguments(tool_calls):
    return {"result": SimpleNamespace(content="", type="ai", tool_calls=tool_calls),
            "exception": None}


class TestAssistantMessageToolCalls(unittest.TestCase):
    """A model can ask for several tools in one response; record them all."""

    def test_every_tool_call_is_recorded(self):
        recorded = json.loads(extract_assistant_message(_arguments([FLIGHT, HOTEL])))
        self.assertEqual([call["name"] for call in recorded["ai"]],
                         ["book_flight", "book_hotel"])

    def test_a_single_tool_call_keeps_its_old_shape(self):
        recorded = json.loads(extract_assistant_message(_arguments([FLIGHT])))
        self.assertEqual(recorded["ai"], FLIGHT)

    def test_no_tool_calls_records_nothing(self):
        self.assertEqual(extract_assistant_message(_arguments([])), "")


class TestToolName(unittest.TestCase):
    """entity.3.name: one name on its own, several as a list."""

    def _name_for(self, tool_calls):
        result = SimpleNamespace(content="", type="ai", tool_calls=tool_calls,
                                 response_metadata={"finish_reason": "tool_calls"})
        return extract_tool_name({"result": result, "exception": None})

    def test_one_tool_call_is_reported_as_a_plain_name(self):
        self.assertEqual(self._name_for([FLIGHT]), "book_flight")

    def test_several_tool_calls_are_all_reported(self):
        self.assertEqual(self._name_for([FLIGHT, HOTEL]), ["book_flight", "book_hotel"])

    def test_no_tool_calls_reports_nothing(self):
        self.assertIsNone(self._name_for([]))

    def test_an_entry_without_a_name_does_not_lose_the_others(self):
        self.assertEqual(self._name_for([FLIGHT, {"args": {}}, HOTEL]),
                         ["book_flight", "book_hotel"])


class TestAgentInferenceType(unittest.TestCase):
    """A delegation counts wherever it sits in the list, not only first."""

    def _type_for(self, tool_calls):
        with patch("monocle_apptrace.instrumentation.metamodel.langchain._helper.get_value",
                   return_value="transfer_to_"):
            return agent_inference_type(_arguments(tool_calls))

    def test_delegation_after_a_plain_tool_call(self):
        handoff = {"name": "transfer_to_hotel_agent", "args": {}, "id": "c3"}
        self.assertEqual(self._type_for([FLIGHT, handoff]), INFERENCE_AGENT_DELEGATION)

    def test_plain_tool_calls_only(self):
        self.assertEqual(self._type_for([FLIGHT, HOTEL]), INFERENCE_TOOL_CALL)

    def test_no_tool_calls_ends_the_turn(self):
        self.assertEqual(self._type_for([]), INFERENCE_TURN_END)


if __name__ == "__main__":
    unittest.main()
