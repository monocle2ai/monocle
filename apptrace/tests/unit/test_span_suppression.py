"""Unit tests for the MONOCLE_SUPPRESS_SPANS context flag.

The flag makes an instrumented method run normally but emit no span at all. It
exists so a test harness can drive its own LLM-backed agent under instrumentation
without polluting the spans of the agent under test. It is distinct from
MONOCLE_SKIP_EXECUTIONS, which skips a tool's *execution* but still emits a span.
"""
import asyncio
import unittest

from common.custom_exporter import CustomConsoleSpanExporter
from common.dummy_class import DummyClass
from opentelemetry.context import attach, detach, set_value
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from monocle_apptrace import setup_monocle_telemetry
from monocle_apptrace.instrumentation.common.constants import MONOCLE_SUPPRESS_SPANS
from monocle_apptrace.instrumentation.common.wrapper import atask_wrapper, task_wrapper
from monocle_apptrace.instrumentation.common.wrapper_method import WrapperMethod


class TestSpanSuppression(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.exporter = CustomConsoleSpanExporter()
        cls.instrumentor = setup_monocle_telemetry(
            workflow_name="span_suppression_test",
            span_processors=[SimpleSpanProcessor(cls.exporter)],
            wrapper_methods=[
                WrapperMethod(
                    package="common.dummy_class",
                    object_name="DummyClass",
                    method="double_it",
                    span_name="double_it",
                    wrapper_method=task_wrapper,
                ),
                WrapperMethod(
                    package="common.dummy_class",
                    object_name="DummyClass",
                    method="add1",
                    span_name="add1",
                    wrapper_method=atask_wrapper,
                ),
            ],
        )
        cls.dummy = DummyClass()

    @classmethod
    def tearDownClass(cls):
        if cls.instrumentor is not None:
            cls.instrumentor.uninstrument()

    def setUp(self):
        self.exporter.reset()

    def tearDown(self):
        self.exporter.reset()

    def test_sync_call_emits_spans_without_suppression(self):
        """Baseline: the instrumented method is traced when the flag is unset."""
        self.assertEqual(self.dummy.double_it(5), 10)
        self.exporter.force_flush()
        self.assertGreater(len(self.exporter.captured_spans), 0)

    def test_sync_call_emits_no_span_under_suppression(self):
        """The method still runs and returns, but nothing is exported."""
        token = attach(set_value(MONOCLE_SUPPRESS_SPANS, True))
        try:
            self.assertEqual(self.dummy.double_it(5), 10)
        finally:
            detach(token)
        self.exporter.force_flush()
        self.assertEqual(self.exporter.captured_spans, [])

    def test_async_call_emits_no_span_under_suppression(self):
        """The flag propagates into the async wrapper via the OTel context."""

        async def run():
            token = attach(set_value(MONOCLE_SUPPRESS_SPANS, True))
            try:
                return await self.dummy.add1(5)
            finally:
                detach(token)

        asyncio.run(run())
        self.exporter.force_flush()
        self.assertEqual(self.exporter.captured_spans, [])

    def test_spans_resume_after_detach(self):
        """Suppression unwinds with the context token; later calls trace again."""
        token = attach(set_value(MONOCLE_SUPPRESS_SPANS, True))
        self.dummy.double_it(5)
        detach(token)
        self.exporter.reset()

        self.dummy.double_it(6)
        self.exporter.force_flush()
        self.assertGreater(len(self.exporter.captured_spans), 0)


if __name__ == "__main__":
    unittest.main()
