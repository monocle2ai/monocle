from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from monocle_apptrace.instrumentation.common import wrapper


def test_non_isolated_mode_yields_span_from_otel_context_manager():
    tracer = MagicMock()
    span = MagicMock()

    @contextmanager
    def current_span(*args, **kwargs):
        yield span

    tracer.start_as_current_span.side_effect = current_span

    with patch.object(wrapper, "ISOLATE_MONOCLE_SPANS", False):
        with wrapper.start_as_monocle_span(
            tracer,
            "inference",
            auto_close_span=True,
            start_time=123,
        ) as current:
            assert current is span

    tracer.start_as_current_span.assert_called_once_with(
        "inference",
        end_on_exit=True,
        start_time=123,
    )


def test_non_isolated_monocle_span_uses_active_otel_parent():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")

    with tracer.start_as_current_span("application") as parent:
        parent_context = parent.get_span_context()
        with patch.object(wrapper, "ISOLATE_MONOCLE_SPANS", False):
            with wrapper.start_as_monocle_span(
                tracer,
                "inference",
                auto_close_span=True,
            ) as child:
                child_context = child.get_span_context()

    spans = {span.name: span for span in exporter.get_finished_spans()}
    assert child_context.trace_id == parent_context.trace_id
    assert spans["inference"].parent.span_id == parent_context.span_id
