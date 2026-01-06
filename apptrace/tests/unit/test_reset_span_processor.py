import typing
import unittest
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from common.dummy_class import dummy_wrapper
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry, reset_span_processors, get_monocle_instrumentor, get_monocle_span_processor
from monocle_apptrace.instrumentation.common.wrapper_method import WrapperMethod
from monocle_apptrace.instrumentation.common.wrapper import atask_wrapper
from common.dummy_class import DummyClass

class Exporter1(InMemorySpanExporter):
    exec_count:int = 0
    def export(self, spans: typing.Sequence[ReadableSpan]) -> SpanExportResult:
        Exporter1.exec_count += 1
        return super().export(spans)


class Exporter2(InMemorySpanExporter):
    exec_count:int = 0
    def export(self, spans: typing.Sequence[ReadableSpan]) -> SpanExportResult:
        Exporter2.exec_count += 1
        return super().export(spans)

class TestHandler(unittest.TestCase):

    instrumentor = None

    def setUp(self):
        # Reset counters before test
        Exporter1.exec_count = 0
        Exporter2.exec_count = 0
        
        # Clean up any existing instrumentor state
        existing_instrumentor = get_monocle_instrumentor()
        if existing_instrumentor is not None:
            try:
                existing_instrumentor.uninstrument()
            except:
                pass
        
        self.exporter1 = Exporter1()
        self.exporter2 = Exporter2()
        
        self.instrumentor = setup_monocle_telemetry(
                workflow_name="reset_test",
                span_processors=[
                        SimpleSpanProcessor(self.exporter1)
                    ],
                                union_with_default_methods=False,
                wrapper_methods=[
                    WrapperMethod(
                        package="common.dummy_class",
                        object_name="DummyClass",
                        method="dummy_method",
                        span_name="dummy.span",
                        wrapper_method=dummy_wrapper)
                ])

    def tearDown(self) -> None:
        try:
            if self.instrumentor is not None:
                self.instrumentor.uninstrument()
        except Exception as e:
            print("Uninstrument failed:", e)
        
        return super().tearDown()

    def test_call_multi_setup_telemetry(self):
        """
        Test that reset span processor.
        """
        assert get_monocle_instrumentor() is self.instrumentor , "get_monocle_instrumentor should return the first instrumentor instance."

        #confirm exporter1 is working
        dummy = DummyClass()
        dummy.dummy_method()
        assert Exporter1.exec_count == 1, f"Exporter1 should have exported one batch of spans. Got: {Exporter1.exec_count}"

        # Now reset the span processors to use exporter2
        reset_span_processors([SimpleSpanProcessor(self.exporter2)])

        #Verify that the instrumentor instance is the same after reset and the span processor has been updated
        instrumentor2 = get_monocle_instrumentor()
        assert self.instrumentor is instrumentor2, "reset_span_processors should not create a new instrumentor instance."
        monocle_span_processor = get_monocle_span_processor()
        assert len(monocle_span_processor._span_processors) == 1, f"There should be one span processor after reset. Got: {len(monocle_span_processor._span_processors)}"
        assert isinstance(monocle_span_processor._span_processors[0], SimpleSpanProcessor), "The span processor should be of type SimpleSpanProcessor."

        #confirm exporter2 is working
        dummy.dummy_method()
       
        assert Exporter2.exec_count == 1, f"Exporter2 should have exported one batch of spans. Got: {Exporter2.exec_count}"
        assert Exporter1.exec_count == 1, f"Exporter1 should not have exported any additional spans. Got: {Exporter1.exec_count}"


if __name__ == '__main__':
    unittest.main()