import logging
import pytest
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from monocle_apptrace.exporters.file_exporter import FileSpanExporter
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor, ConsoleSpanExporter


@pytest.fixture(autouse=True)
def cleanup_telemetry_state():
    """Reset telemetry state before and after each test."""
    yield
    # Cleanup after test
    try:
        from monocle_apptrace.instrumentation.common.instrumentor import get_monocle_instrumentor
        instrumentor = get_monocle_instrumentor()
        if instrumentor is not None and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()
    except Exception:
        pass


def test_multiple_setup_monocle_telemetry_calls(caplog):
    caplog.set_level(logging.WARNING)

    instrumentor = setup_monocle_telemetry(
        workflow_name="duplicate_setup_test",
        monocle_exporters_list="console,memory",
        union_with_default_methods="true",
    )

    duplicate_same = setup_monocle_telemetry(
        workflow_name="duplicate_setup_test",
        monocle_exporters_list="memory,console",
        union_with_default_methods=True,
    )
    assert duplicate_same is instrumentor

    duplicate_changed = setup_monocle_telemetry(
        workflow_name="duplicate_setup_test",
        monocle_exporters_list="console,file",
        union_with_default_methods=False,
    )
    assert duplicate_changed is instrumentor

    messages = [record.getMessage() for record in caplog.records]

    duplicate_warnings = [
        message
        for message in messages
        if "Ignoring duplicate setup_monocle_telemetry() call" in message
    ]
    assert len(duplicate_warnings) >= 2

    diff_warnings = [
        message
        for message in messages
        if "configuration differences" in message.lower()
    ]
    assert len(diff_warnings) == 1

    latest_diff_warning = diff_warnings[-1]
    assert "union_with_default_methods" in latest_diff_warning
    assert "monocle_exporters_list" in latest_diff_warning
    assert "previous" in latest_diff_warning
    assert "current" in latest_diff_warning


def test_multiple_setup_with_span_processors(caplog):
    caplog.set_level(logging.WARNING)

    file_exporter = FileSpanExporter()
    console_exporter = ConsoleSpanExporter()
    
    span_processors = [
        BatchSpanProcessor(file_exporter),
        SimpleSpanProcessor(console_exporter)
    ]
    
    instrumentor = setup_monocle_telemetry(
        workflow_name="span_processors_test",
        span_processors=span_processors,
    )

    # Call again with same processor types (should not show config differences)
    duplicate_same = setup_monocle_telemetry(
        workflow_name="span_processors_test",
        span_processors=[
            BatchSpanProcessor(FileSpanExporter()),
            SimpleSpanProcessor(ConsoleSpanExporter())
        ],
    )
    assert duplicate_same is instrumentor

    # Call with different processor types (should show config differences)
    duplicate_changed = setup_monocle_telemetry(
        workflow_name="span_processors_test",
        span_processors=[
            SimpleSpanProcessor(FileSpanExporter()),
        ],
    )
    assert duplicate_changed is instrumentor

    messages = [record.getMessage() for record in caplog.records]

    duplicate_warnings = [
        message
        for message in messages
        if "Ignoring duplicate setup_monocle_telemetry() call" in message
    ]
    assert len(duplicate_warnings) >= 2

    diff_warnings = [
        message
        for message in messages
        if "configuration differences" in message.lower()
    ]
    assert len(diff_warnings) == 1

    latest_diff_warning = diff_warnings[-1]
    assert "span_processors" in latest_diff_warning
    assert "previous" in latest_diff_warning
    assert "current" in latest_diff_warning
