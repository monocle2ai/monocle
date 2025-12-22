import typing
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

def test_call_multi_setup_telemetry():
    """
    Test that reset span processor.
    """
    exporter1 = Exporter1()
    exporter2 = Exporter2()
    instrumentor1 = setup_monocle_telemetry(
            workflow_name="reset_test",
            span_processors=[
                    SimpleSpanProcessor(exporter1)
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
    assert get_monocle_instrumentor() is instrumentor1 , "get_monocle_instrumentor should return the first instrumentor instance."

    #confirm exporter1 is working
    dummy = DummyClass()
    dummy.dummy_method()
    assert Exporter1.exec_count == 1, "Exporter1 should have exported one batch of spans."

    # Now reset the span processors to use exporter2
    reset_span_processors([SimpleSpanProcessor(exporter2)])

    #Verify that the instrumentor instance is the same after reset and the span processor has been updated
    instrumentor2 = get_monocle_instrumentor()
    assert instrumentor1 is instrumentor2, "reset_span_processors should not create a new instrumentor instance."
    monocle_span_processor = get_monocle_span_processor()
    assert len(monocle_span_processor._span_processors) == 1, "There should be one span processor after reset."
    assert isinstance(monocle_span_processor._span_processors[0], SimpleSpanProcessor), "The span processor should be of type SimpleSpanProcessor."

    #confirm exporter2 is working
    dummy.dummy_method()
    assert Exporter2.exec_count == 1, "Exporter2 should have exported one batch of spans."
    assert Exporter1.exec_count == 1, "Exporter1 should not have exported any additional spans."