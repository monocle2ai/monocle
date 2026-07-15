import unittest
import uuid
from unittest.mock import MagicMock, patch
from opentelemetry.sdk.trace import Span

from monocle_apptrace.instrumentation.common.span_handler import SpanHandler


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


if __name__ == "__main__":
    unittest.main()
