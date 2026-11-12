import os
import unittest
from unittest.mock import MagicMock, patch

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from monocle_apptrace.exporters.paygentic.paygentic_exporter import PaygenticSpanExporter

POST = "monocle_apptrace.exporters.paygentic.paygentic_exporter.requests.post"


def _make_span(scopes):
    """Finish one inference span carrying the given monocle scopes and return it."""
    memory = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(memory))
    tracer = provider.get_tracer(__name__)
    span = tracer.start_span("openai.chat")
    span.set_attribute("monocle_apptrace.version", "0.8.0")
    span.set_attribute("span.type", "inference")
    span.set_attribute("entity.1.type", "inference.openai")
    for name, value in scopes.items():
        span.set_attribute(f"scope.{name}", value)
    span.add_event("metadata", {"total_tokens": 42})
    span.end()
    return memory.get_finished_spans()[0]


class TestPaygenticCustomerScopes(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {"PAYGENTIC_API_KEY": "test-key"})
        env.start()
        self.addCleanup(env.stop)
        self.exporter = PaygenticSpanExporter(endpoint="https://paygentic.test/v0/events")

    def _export(self, scopes):
        """Export one span and return the event bodies posted to Paygentic."""
        with patch(POST, return_value=MagicMock(status_code=202)) as post:
            self.exporter.export([_make_span(scopes)])
        return [call.kwargs["json"] for call in post.call_args_list]

    def test_customer_id_is_sent_as_subject(self):
        (event,) = self._export({"customerId": "cus_123"})
        self.assertEqual(event["subject"], "cus_123")
        self.assertNotIn("externalSubject", event)

    def test_external_customer_id_is_sent_as_external_subject(self):
        (event,) = self._export({"externalCustomerId": "acme-42"})
        self.assertEqual(event["externalSubject"], "acme-42")
        self.assertNotIn("subject", event)

    def test_both_customer_scopes_are_sent(self):
        (event,) = self._export({"customerId": "cus_123", "externalCustomerId": "acme-42"})
        self.assertEqual(event["subject"], "cus_123")
        self.assertEqual(event["externalSubject"], "acme-42")

    def test_subscription_id_is_not_the_subject(self):
        (event,) = self._export({"subscriptionId": "sub_456", "customerId": "cus_123"})
        self.assertEqual(event["subject"], "cus_123")

    def test_span_without_a_customer_scope_is_dropped(self):
        with self.assertLogs("monocle_apptrace.exporters.paygentic.paygentic_exporter", "ERROR"):
            events = self._export({"subscriptionId": "sub_456"})
        self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main()
