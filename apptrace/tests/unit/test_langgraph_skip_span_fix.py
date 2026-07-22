"""
Unit tests for LangGraph skip_span fix preventing duplicate agentic.invocation spans.

The fix: skip_span() skips stream/astream span creation when already in an
agentic.invocation scope (i.e., when ainvoke internally calls astream).
"""

import types
import unittest
from unittest.mock import MagicMock, patch

from monocle_apptrace.instrumentation.common.scope_wrapper import start_scope, stop_scope
from monocle_apptrace.instrumentation.common.constants import (
    AGENT_INVOCATION_SPAN_NAME,
    AGENT_REQUEST_SPAN_NAME,
)
from monocle_apptrace.instrumentation.metamodel.langgraph.langgraph_processor import (
    LanggraphAgentHandler,
)
from monocle_apptrace.instrumentation.metamodel.langgraph.methods import LANGGRAPH_METHODS


def _fake_graph():
    """A CompiledStateGraph-shaped stand-in."""
    graph = types.SimpleNamespace()
    graph.builder = types.SimpleNamespace(nodes={"agent": None})
    graph.name = "test_graph"
    return graph


class TestLanggraphSkipSpanFix(unittest.TestCase):
    """Test that skip_span prevents duplicate agentic.invocation spans."""

    def setUp(self):
        self.handler = LanggraphAgentHandler()
        self.instance = _fake_graph()
        self.mock_wrapped = MagicMock()
        self.mock_args = ()
        self.mock_kwargs = {}

    def _to_wrap(self, method):
        """Get the instrumentation config for a method."""
        return dict(next(m for m in LANGGRAPH_METHODS if m["method"] == method))

    def test_astream_skipped_when_in_agentic_invocation_scope(self):
        """Test that astream() skips span creation when already in agentic.invocation scope."""
        to_wrap = self._to_wrap("astream")
        
        # Simulate being inside an agentic.invocation scope (as when ainvoke calls astream)
        scope_token = start_scope(AGENT_INVOCATION_SPAN_NAME, "test-invocation-id")
        try:
            # skip_span should return True to prevent duplicate span
            result = self.handler.skip_span(
                to_wrap, self.mock_wrapped, self.instance, self.mock_args, self.mock_kwargs
            )
            self.assertTrue(
                result,
                "skip_span should return True for astream() when in agentic.invocation scope"
            )
        finally:
            stop_scope(scope_token)

    def test_astream_not_skipped_when_no_scope(self):
        """Test that astream() creates span normally when called directly (no existing scope)."""
        to_wrap = self._to_wrap("astream")
        
        # No agentic.invocation scope set - direct astream() call
        result = self.handler.skip_span(
            to_wrap, self.mock_wrapped, self.instance, self.mock_args, self.mock_kwargs
        )
        self.assertFalse(
            result,
            "skip_span should return False for direct astream() call (no existing scope)"
        )

    def test_stream_skipped_when_in_agentic_invocation_scope(self):
        """Test that stream() skips span creation when already in agentic.invocation scope."""
        to_wrap = self._to_wrap("stream")
        
        # Simulate being inside an agentic.invocation scope (as when invoke calls stream)
        scope_token = start_scope(AGENT_INVOCATION_SPAN_NAME, "test-invocation-id")
        try:
            result = self.handler.skip_span(
                to_wrap, self.mock_wrapped, self.instance, self.mock_args, self.mock_kwargs
            )
            self.assertTrue(
                result,
                "skip_span should return True for stream() when in agentic.invocation scope"
            )
        finally:
            stop_scope(scope_token)

    def test_stream_not_skipped_when_no_scope(self):
        """Test that stream() creates span normally when called directly."""
        to_wrap = self._to_wrap("stream")
        
        # No agentic.invocation scope set - direct stream() call
        result = self.handler.skip_span(
            to_wrap, self.mock_wrapped, self.instance, self.mock_args, self.mock_kwargs
        )
        self.assertFalse(
            result,
            "skip_span should return False for direct stream() call (no existing scope)"
        )

    def test_ainvoke_not_skipped(self):
        """Test that ainvoke() is never skipped (it's the primary entry point)."""
        to_wrap = self._to_wrap("ainvoke")
        
        # Even if there's an existing scope, ainvoke should not be skipped by this logic
        scope_token = start_scope(AGENT_INVOCATION_SPAN_NAME, "existing-scope")
        try:
            result = self.handler.skip_span(
                to_wrap, self.mock_wrapped, self.instance, self.mock_args, self.mock_kwargs
            )
            # ainvoke is not in the ["stream", "astream"] list, so skip_span returns False
            # (or whatever the parent class returns)
            self.assertFalse(
                result,
                "skip_span should not skip ainvoke() via this fix logic"
            )
        finally:
            stop_scope(scope_token)

    def test_invoke_not_skipped(self):
        """Test that invoke() is never skipped."""
        to_wrap = self._to_wrap("invoke")
        
        scope_token = start_scope(AGENT_INVOCATION_SPAN_NAME, "existing-scope")
        try:
            result = self.handler.skip_span(
                to_wrap, self.mock_wrapped, self.instance, self.mock_args, self.mock_kwargs
            )
            self.assertFalse(
                result,
                "skip_span should not skip invoke() via this fix logic"
            )
        finally:
            stop_scope(scope_token)

    def test_agentic_turn_scope_also_triggers_skip(self):
        """Test that stream/astream are skipped when already in agentic.turn scope."""
        to_wrap = self._to_wrap("astream")
        
        scope_token = start_scope(AGENT_REQUEST_SPAN_NAME, "some-turn-id")
        try:
            result = self.handler.skip_span(
                to_wrap, self.mock_wrapped, self.instance, self.mock_args, self.mock_kwargs
            )
            self.assertTrue(
                result,
                "skip_span should skip astream() when agentic.turn scope is set"
            )
        finally:
            stop_scope(scope_token)


class TestLanggraphSkipSpanIntegration(unittest.TestCase):
    """Integration-style tests simulating the full invoke->stream call chain."""

    def setUp(self):
        self.handler = LanggraphAgentHandler()
        self.instance = _fake_graph()

    def _to_wrap(self, method):
        return dict(next(m for m in LANGGRAPH_METHODS if m["method"] == method))

    def test_ainvoke_then_astream_sequence(self):
        """
        Simulate the actual call sequence:
        1. ainvoke() is called -> creates scope
        2. ainvoke() internally calls astream()
        3. astream()'s skip_span should return True
        """
        # Step 1: ainvoke creates a scope
        ainvoke_to_wrap = self._to_wrap("ainvoke")
        scope_token = start_scope(AGENT_INVOCATION_SPAN_NAME, "ainvoke-scope-123")
        
        try:
            # Step 2: During ainvoke's execution, astream() is called
            astream_to_wrap = self._to_wrap("astream")
            
            # Step 3: astream's skip_span should detect the existing scope and skip
            should_skip = self.handler.skip_span(
                astream_to_wrap, None, self.instance, (), {}
            )
            
            self.assertTrue(
                should_skip,
                "When ainvoke calls astream internally, astream should skip span creation"
            )
        finally:
            stop_scope(scope_token)

    def test_direct_astream_call_creates_span(self):
        """
        When user directly calls astream() (not via ainvoke), it should create a span.
        """
        astream_to_wrap = self._to_wrap("astream")
        
        # No scope exists - direct call
        should_skip = self.handler.skip_span(
            astream_to_wrap, None, self.instance, (), {}
        )
        
        self.assertFalse(
            should_skip,
            "Direct astream() call should create a span normally"
        )


if __name__ == "__main__":
    unittest.main()
