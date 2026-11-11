import unittest
from unittest.mock import MagicMock, patch

from opentelemetry.sdk.trace import Span
from opentelemetry.trace import SpanContext

from monocle_apptrace.instrumentation.common.span_handler import SpanHandler


class TestWorkflowRemoteParent(unittest.TestCase):
    """The workflow span is flagged when its parent came from another process."""

    def _workflow_span(self, parent):
        span = MagicMock(spec=Span)
        span.attributes = {}
        span.set_attribute = MagicMock(side_effect=lambda k, v: span.attributes.__setitem__(k, v))
        span.parent = parent
        with patch.object(SpanHandler, "get_workflow_name", return_value="a2a_travel_agent"):
            SpanHandler.set_workflow_attributes(None, span)
        return span

    def test_remote_parent_is_flagged(self):
        parent = SpanContext(trace_id=1, span_id=2, is_remote=True)
        self.assertTrue(self._workflow_span(parent).attributes.get("parent.is_remote"))

    def test_local_parent_is_not_flagged(self):
        parent = SpanContext(trace_id=1, span_id=2, is_remote=False)
        self.assertNotIn("parent.is_remote", self._workflow_span(parent).attributes)

    def test_no_parent_is_not_flagged(self):
        self.assertNotIn("parent.is_remote", self._workflow_span(None).attributes)


if __name__ == "__main__":
    unittest.main()
