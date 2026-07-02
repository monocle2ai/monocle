import unittest

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompt_values import ChatPromptValue

from monocle_apptrace.instrumentation.metamodel.langchain._helper import extract_messages


class TestExtractMessages(unittest.TestCase):
    """`extract_messages` receives the positional args of BaseChatModel.invoke/ainvoke,
    i.e. args[0] is the model input, which LangChain accepts as a str, a PromptValue,
    or a list of BaseMessage."""

    def test_string_input(self):
        # llm.invoke("...") / with_structured_output(...).ainvoke("...")
        self.assertEqual(extract_messages(("say ok in one word",)), ["say ok in one word"])

    def test_string_input_not_iterated_per_character(self):
        # Regression: a str must not fall through to per-character iteration (which
        # yielded an empty result and dropped data.input entirely).
        self.assertNotEqual(extract_messages(("hello",)), [])

    def test_message_list(self):
        out = extract_messages(([SystemMessage(content="You are terse."),
                                 HumanMessage(content="Reply blue.")],))
        self.assertIn('{"system": "You are terse."}', out)
        self.assertIn('{"human": "Reply blue."}', out)

    def test_prompt_value(self):
        out = extract_messages((ChatPromptValue(messages=[SystemMessage(content="sys"),
                                                          HumanMessage(content="hi")]),))
        self.assertIn('{"system": "sys"}', out)
        self.assertIn('{"human": "hi"}', out)

    def test_dict_messages(self):
        # LangChain accepts OpenAI-style dict messages: [{"role", "content"}]
        out = extract_messages(([{"role": "system", "content": "sys"},
                                 {"role": "user", "content": "hi"}],))
        self.assertIn('{"system": "sys"}', out)
        self.assertIn('{"user": "hi"}', out)

    def test_dict_messages_not_dropped(self):
        # Regression: dict messages must not be silently dropped (empty result).
        self.assertNotEqual(extract_messages(([{"role": "user", "content": "hi"}],)), [])

    def test_tool_call_message(self):
        msg = AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "1"}])
        out = extract_messages(([msg],))
        self.assertTrue(out and "tool_call" in out[0])

    def test_empty_args(self):
        self.assertEqual(extract_messages(()), [])


if __name__ == "__main__":
    unittest.main()
