"""MONOCLE_SUPPRESS_SPANS: an instrumented call still runs, but emits no span while the flag is set.

Scenario tests set it around their own simulated user and judge, so traces hold only the agent under test.
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
                WrapperMethod(package="common.dummy_class", object_name="DummyClass", method="double_it",
                              span_name="double_it", wrapper_method=task_wrapper),
                WrapperMethod(package="common.dummy_class", object_name="DummyClass", method="add1",
                              span_name="add1", wrapper_method=atask_wrapper),
            ])
        cls.dummy = DummyClass()

    @classmethod
    def tearDownClass(cls):
        cls.instrumentor.uninstrument()

    def setUp(self):
        self.exporter.reset()

    def spans(self):
        self.exporter.force_flush()
        return self.exporter.captured_spans

    def test_sync_and_async_calls_run_but_emit_nothing_while_suppressed(self):
        async def add1():
            return await self.dummy.add1(5)

        token = attach(set_value(MONOCLE_SUPPRESS_SPANS, True))
        try:
            self.assertEqual(self.dummy.double_it(5), 10)
            self.assertEqual(asyncio.run(add1()), 11)
        finally:
            detach(token)
        self.assertEqual(self.spans(), [])

    def test_spans_come_back_once_the_flag_is_detached(self):
        self.dummy.double_it(5)
        self.assertGreater(len(self.spans()), 0)


if __name__ == "__main__":
    unittest.main()
