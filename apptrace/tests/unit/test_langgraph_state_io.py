import dataclasses
import json
import unittest

from monocle_apptrace.instrumentation.metamodel.langgraph import _helper
from monocle_apptrace.instrumentation.metamodel.langgraph.langgraph_stream_processor import (
    LanggraphStreamProcessor,
)
from monocle_apptrace.instrumentation.common.stream_processor import StreamState


class _Msg:
    def __init__(self, content, type="ai"):
        self.content = content
        self.type = type


@dataclasses.dataclass
class _DataclassState:
    messages: list
    tool_iterations: int = 0


class _PydanticLikeState:
    """Stands in for a Pydantic BaseModel-based StateGraph schema (v1 or v2)."""

    model_fields = {"messages", "task"}

    def __init__(self, messages=None, task=None):
        self.messages = messages or []
        self.task = task


class TestLanggraphMessageStateIO(unittest.TestCase):
    """Message-based graphs keep extracting message text (no behavior change)."""

    def test_message_input_preferred(self):
        state = {"messages": [_Msg("hello", type="human")]}
        out = _helper.extract_agent_input({"kwargs": {"input": state}, "args": []})
        self.assertEqual(json.loads(out), ["hello"])

    def test_message_output_preferred(self):
        state = {"messages": [_Msg("the answer")]}
        self.assertEqual(_helper.extract_agent_response(state), "the answer")


class TestLanggraphCustomStateIO(unittest.TestCase):
    """Custom (non message-based) StateGraphs must still carry their I/O, serialized."""

    def test_custom_input_serialized(self):
        state = {"task": {"query": "Is AI in a hype cycle?"}}
        out = _helper.extract_agent_input({"kwargs": {"input": state}, "args": []})
        self.assertEqual(json.loads(out), state)

    def test_custom_input_from_positional_args(self):
        state = {"task": {"query": "q"}}
        out = _helper.extract_agent_input({"kwargs": {}, "args": [state]})
        self.assertEqual(json.loads(out), state)

    def test_custom_output_serialized(self):
        state = {"task": {"query": "q"}, "report": "final report text"}
        self.assertEqual(json.loads(_helper.extract_agent_response(state)), state)

    def test_empty_state_returns_empty(self):
        self.assertEqual(_helper.extract_agent_input({"kwargs": {"input": {}}, "args": []}), "")
        self.assertEqual(_helper.extract_agent_response({}), "")


class TestLanggraphDataclassAndPydanticStateIO(unittest.TestCase):
    """Custom StateGraphs commonly use a dataclass or Pydantic BaseModel schema instead of a
    TypedDict. A nested sub-agent graph called with such a state (e.g. `self._graph.ainvoke(state)`
    where state is a @dataclass) must still have its human message / state extracted, not silently
    dropped because extract_agent_input only recognized plain dicts."""

    def test_dataclass_state_message_extracted(self):
        state = _DataclassState(messages=[_Msg("hello", type="human")])
        out = _helper.extract_agent_input({"kwargs": {}, "args": [state]})
        self.assertEqual(json.loads(out), ["hello"])

    def test_pydantic_like_state_message_extracted(self):
        state = _PydanticLikeState(messages=[_Msg("hello", type="human")])
        out = _helper.extract_agent_input({"kwargs": {}, "args": [state]})
        self.assertEqual(json.loads(out), ["hello"])

    def test_dataclass_state_without_human_message_serialized(self):
        state = _DataclassState(messages=[_Msg("assistant reply", type="ai")], tool_iterations=2)
        out = _helper.extract_agent_input({"kwargs": {}, "args": [state]})
        self.assertEqual(json.loads(out), ["assistant reply"])

    def test_plain_object_state_returns_empty_not_crash(self):
        class _NotAState:
            pass

        out = _helper.extract_agent_input({"kwargs": {}, "args": [_NotAState()]})
        self.assertEqual(out, "")


class TestLanggraphStreamFallback(unittest.TestCase):
    """astream of a custom StateGraph yields plain state dicts (no message channel);
    the stream processor must fall back to the last state snapshot."""

    def _run(self, items):
        proc = LanggraphStreamProcessor()
        state = StreamState()
        for item in items:
            proc.process_fragment(item, state)
        proc.assemble_data(state)
        return state.accumulated_response

    def test_values_mode_snapshot(self):
        chunks = [
            {"task": {"query": "q"}},
            {"task": {"query": "q"}, "report": "done"},
        ]
        out = self._run(chunks)
        self.assertEqual(json.loads(out), chunks[-1])

    def test_message_chunks_still_win(self):
        chunks = [{"messages": [_Msg("streamed answer")]}]
        out = self._run(chunks)
        self.assertEqual(out, "streamed answer")

    def test_tuple_values_mode_snapshot(self):
        chunks = [("values", {"task": {"query": "q"}, "report": "final"})]
        out = self._run(chunks)
        self.assertEqual(json.loads(out), {"task": {"query": "q"}, "report": "final"})


if __name__ == "__main__":
    unittest.main()
