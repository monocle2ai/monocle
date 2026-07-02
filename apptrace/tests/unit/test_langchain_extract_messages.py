import json
import unittest
from types import SimpleNamespace

from monocle_apptrace.instrumentation.metamodel.langchain._helper import extract_messages


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


if __name__ == "__main__":
    unittest.main()
