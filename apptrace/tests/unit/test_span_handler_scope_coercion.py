import unittest
import uuid
from unittest.mock import MagicMock, patch
from opentelemetry.sdk.trace import Span

from monocle_apptrace.instrumentation.common.span_handler import SpanHandler
from monocle_apptrace.instrumentation.common.constants import AGENT_REQUEST_SPAN_NAME
from monocle_apptrace.instrumentation.common.utils import get_scopes, set_scopes, remove_scopes
from monocle_apptrace.instrumentation.metamodel.adk import adk_handler


class TestScopeValueCoercion(unittest.TestCase):
    """Scope values derived from app objects (e.g. a UUID thread_id) must be coerced to
    OTEL-valid attribute types, otherwise OTEL drops them and the run loses its session anchor."""

    def test_primitives_pass_through(self):
        for value in ["s", b"b", 1, 1.5, True]:
            self.assertIs(SpanHandler._coerce_scope_value(value), value)

    def test_primitive_sequences_pass_through(self):
        value = ["a", "b"]
        self.assertIs(SpanHandler._coerce_scope_value(value), value)

    def test_uuid_is_stringified(self):
        u = uuid.uuid4()
        coerced = SpanHandler._coerce_scope_value(u)
        self.assertEqual(coerced, str(u))
        self.assertIsInstance(coerced, str)

    def test_object_is_stringified(self):
        self.assertEqual(SpanHandler._coerce_scope_value({"a": 1}), str({"a": 1}))

    def test_default_attributes_stringify_uuid_scope(self):
        attrs = {}
        span = MagicMock(spec=Span)
        span.set_attribute = MagicMock(side_effect=lambda k, v: attrs.__setitem__(k, v))
        u = uuid.uuid4()
        with patch(
            "monocle_apptrace.instrumentation.common.span_handler.get_scopes",
            return_value={"agentic.session": u},
        ), patch.object(SpanHandler, "get_workflow_name", return_value=None):
            SpanHandler.set_default_monocle_attributes(span)
        self.assertEqual(attrs["scope.agentic.session"], str(u))


class TestAdkTurnId(unittest.TestCase):
    """AdkSpanHandler reuses ADK's invocation id as the turn id, set once on the root agent
    and stamped on every span in the turn (the turn span itself via a parent backfill)."""

    class _Agent:  # a non-Runner, non-orchestrator agent instance
        pass

    class _Ctx:  # ADK's InvocationContext passed to BaseAgent.run_async
        def __init__(self, invocation_id):
            self.invocation_id = invocation_id

    def setUp(self):
        self.handler = adk_handler.AdkSpanHandler()
        self._tokens = []

    def tearDown(self):
        for token in reversed(self._tokens):
            remove_scopes(token)

    def _pre_tracing(self, invocation_id, parent_attrs):
        parent_span = MagicMock(spec=Span)
        parent_span.set_attribute = MagicMock(side_effect=lambda k, v: parent_attrs.__setitem__(k, v))
        with patch.object(adk_handler, "get_parent_span", return_value=parent_span):
            token, _ = self.handler.pre_tracing({}, None, self._Agent(), (self._Ctx(invocation_id),), {})
        if token:
            self._tokens.append(token)
        return dict(get_scopes())

    def test_root_agent_reuses_invocation_id_as_turn_id(self):
        parent_attrs = {}
        scopes = self._pre_tracing("e-abc", parent_attrs)
        self.assertEqual(scopes[AGENT_REQUEST_SPAN_NAME], "e-abc")
        self.assertEqual(parent_attrs[f"scope.{AGENT_REQUEST_SPAN_NAME}"], "e-abc")

    def test_nested_agent_keeps_existing_turn_id(self):
        self._tokens.append(set_scopes({AGENT_REQUEST_SPAN_NAME: "e-outer"}))
        parent_attrs = {}
        scopes = self._pre_tracing("e-inner", parent_attrs)
        self.assertEqual(scopes[AGENT_REQUEST_SPAN_NAME], "e-outer")
        self.assertNotIn(f"scope.{AGENT_REQUEST_SPAN_NAME}", parent_attrs)


if __name__ == "__main__":
    unittest.main()
