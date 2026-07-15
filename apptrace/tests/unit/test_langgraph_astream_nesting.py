import types
import unittest

from monocle_apptrace.instrumentation.common.scope_wrapper import start_scope, stop_scope
from monocle_apptrace.instrumentation.metamodel.langgraph.entities.inference import (
    AGENT,
    AGENT_REQUEST,
    AGENT_REQUEST_STREAM,
    AGENT_STREAM,
)
from monocle_apptrace.instrumentation.metamodel.langgraph.langgraph_processor import (
    LanggraphAgentHandler,
)
from monocle_apptrace.instrumentation.metamodel.langgraph.methods import LANGGRAPH_METHODS


def _fake_graph():
    """A CompiledStateGraph-shaped stand-in that looks like a single-agent instance."""
    graph = types.SimpleNamespace()
    graph.builder = types.SimpleNamespace(nodes={"agent": None})
    graph.name = "test_graph"
    return graph


class TestLanggraphAstreamRegistryDefault(unittest.TestCase):
    """astream's registry default must be invocation-only, matching invoke/ainvoke/astream_events.

    Regression guard: a graph's own ainvoke() commonly delegates internally to astream()
    (or a genuinely nested sub-agent graph is invoked via astream()). pre_tracing() falls back
    to this registry default whenever it detects we're already inside an agentic turn/invocation
    scope (is_scope_set() True). If that default escalates to [turn, invocation] unconditionally
    (as it used to), every such nested call emits a redundant extra 'agentic.turn' span.
    """

    def test_astream_default_is_single_invocation_processor(self):
        entry = next(m for m in LANGGRAPH_METHODS if m["method"] == "astream")
        self.assertNotIn("output_processor_list", entry)
        self.assertIs(entry.get("output_processor"), AGENT_STREAM)

    def test_invoke_and_ainvoke_default_is_single_invocation_processor(self):
        for method in ("invoke", "ainvoke"):
            entry = next(m for m in LANGGRAPH_METHODS if m["method"] == method)
            self.assertNotIn("output_processor_list", entry)
            self.assertIs(entry.get("output_processor"), AGENT)


class TestLanggraphPreTracingNesting(unittest.TestCase):
    """pre_tracing() must only escalate to a [turn, invocation] pair for the outermost call."""

    def setUp(self):
        self.handler = LanggraphAgentHandler()
        self.instance = _fake_graph()

    def _to_wrap(self, method):
        return dict(next(m for m in LANGGRAPH_METHODS if m["method"] == method))

    def test_outermost_astream_escalates_to_turn_and_invocation(self):
        token, alternate = self.handler.pre_tracing(self._to_wrap("astream"), None, self.instance, (), {})
        try:
            self.assertIsNotNone(alternate)
            self.assertEqual(
                alternate["output_processor_list"], [AGENT_REQUEST_STREAM, AGENT_STREAM]
            )
        finally:
            stop_scope(token)

    def test_nested_astream_does_not_reescalate(self):
        scope_name = AGENT_REQUEST.get("type")
        outer_token = start_scope(scope_name, scope_value="already-in-a-turn")
        try:
            _, alternate = self.handler.pre_tracing(self._to_wrap("astream"), None, self.instance, (), {})
            # Falls back to the registry default (single AGENT_STREAM processor) -> no
            # redundant 'turn' span for this nested call.
            self.assertIsNone(alternate)
        finally:
            stop_scope(outer_token)


if __name__ == "__main__":
    unittest.main()
