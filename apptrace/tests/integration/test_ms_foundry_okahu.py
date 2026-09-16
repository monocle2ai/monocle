"""An MS Agent on Azure AI Foundry: correct traces, exported to Okahu.

Covers the two things that were broken for Foundry-hosted agents:

* the workflow span identifies Foundry as the host, rather than reporting
  ``app_hosting.generic``;
* the inference span carries the model's reply and a subtype. FoundryChatClient
  returns openai's raw-response wrapper, which used to leave ``data.output``
  empty and the subtype unset.

``test_foundry_traces_reach_okahu`` then exports a trace and reads it back, which
is what proves the push actually landed.

Requires FOUNDRY_PROJECT_ENDPOINT and Azure credentials (DefaultAzureCredential,
so `az login` or a service principal). The Okahu test additionally needs
OKAHU_API_KEY plus the workflow/app pair to use — see MONOCLE_OKAHU_TEST_* below.
"""

import asyncio
import os
import time
from typing import Annotated

import pytest
from monocle_apptrace.exporters.base_exporter import MonocleInMemorySpanExporter
from monocle_apptrace.instrumentation.common.constants import SPAN_TYPES
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor

try:
    from agent_framework.foundry import FoundryChatClient
    from azure.identity import DefaultAzureCredential

    FOUNDRY_AVAILABLE = True
except ImportError:
    FOUNDRY_AVAILABLE = False

PROJECT_ENDPOINT = os.getenv("FOUNDRY_PROJECT_ENDPOINT")
MODEL = os.getenv("AZURE_OPENAI_MODEL_DEPLOYMENT_NAME") or os.getenv(
    "AZURE_OPENAI_API_DEPLOYMENT"
)

# Okahu attributes an incoming trace by the workflow name the agent reports, but
# the read APIs are keyed by the application name, and the two are mapped
# server-side rather than being the same string. Both are therefore supplied
# rather than derived from each other.
OKAHU_WORKFLOW = os.getenv("MONOCLE_OKAHU_TEST_WORKFLOW")
OKAHU_APP = os.getenv("MONOCLE_OKAHU_TEST_APP")

requires_foundry = pytest.mark.skipif(
    not (FOUNDRY_AVAILABLE and PROJECT_ENDPOINT and MODEL),
    reason="agent-framework-foundry, FOUNDRY_PROJECT_ENDPOINT or a model deployment missing",
)


def book_flight(
    from_airport: Annotated[str, "Departure airport code, e.g. BOM"],
    to_airport: Annotated[str, "Destination airport code, e.g. JFK"],
) -> str:
    """Book a flight between two airports."""
    return f"FLIGHT CONFIRMED: {from_airport} to {to_airport}"


def build_foundry_agent():
    client = FoundryChatClient(
        project_endpoint=PROJECT_ENDPOINT,
        model=MODEL,
        credential=DefaultAzureCredential(),
    )
    return client.as_agent(
        name="Foundry_Flight_Agent",
        instructions="Book flights using book_flight and report the confirmation.",
        tools=[book_flight],
    )


@pytest.fixture(autouse=True)
def reset_telemetry_between_tests():
    """Clear Monocle's global tracing state before each test in this module.

    Telemetry is process-global and setting it up a second time after
    uninstrument() does not re-arm it, so without this the second test onward
    records nothing. The shared conftest fixture only resets per module.
    """
    _reset_monocle_telemetry()
    yield
    _reset_monocle_telemetry()


def _reset_monocle_telemetry():
    from opentelemetry import trace
    from opentelemetry.util._once import Once
    from monocle_apptrace.instrumentation.common.instrumentor import (
        get_monocle_instrumentor,
        set_monocle_instrumentor,
        set_monocle_setup_signature,
        set_monocle_span_processor,
        set_tracer_provider,
    )

    instrumentor = get_monocle_instrumentor()
    if instrumentor is not None and instrumentor.is_instrumented_by_opentelemetry:
        try:
            instrumentor.uninstrument()
        except Exception:
            pass
    set_monocle_instrumentor(None)
    set_monocle_setup_signature(None)
    set_monocle_span_processor(None)
    set_tracer_provider(None)
    # set_tracer_provider() only takes effect once per process; reset the guard so
    # the next test can install its own provider.
    trace._TRACER_PROVIDER = None
    trace._TRACER_PROVIDER_SET_ONCE = Once()


def _instrument(workflow_name, with_okahu=False):
    """Instrument for one test and return (exporter, instrumentor)."""
    memory_exporter = MonocleInMemorySpanExporter()
    processors = [SimpleSpanProcessor(memory_exporter)]
    if with_okahu:
        from monocle_apptrace.exporters.okahu.okahu_exporter import OkahuSpanExporter

        processors.append(BatchSpanProcessor(OkahuSpanExporter()))
    instrumentor = setup_monocle_telemetry(
        workflow_name=workflow_name, span_processors=processors
    )
    return memory_exporter, instrumentor


def _spans_of_type(spans, span_type):
    return [s for s in spans if s.attributes.get("span.type") == span_type]


def _output_event(span):
    for event in span.events:
        if event.name == "data.output":
            return dict(event.attributes or {})
    return {}


def _assert_foundry_trace(spans, source):
    """Assert the attributes this patch fixes, wherever the spans came from."""
    workflow_spans = _spans_of_type(spans, "workflow")
    assert workflow_spans, f"[{source}] expected a workflow span"
    hosting = workflow_spans[0].attributes.get("entity.2.type")
    assert hosting == "app_hosting.azure_ai_foundry", (
        f"[{source}] Foundry hosting not detected, got {hosting!r}; "
        "FOUNDRY_PROJECT_ENDPOINT must be recognised as a Foundry marker"
    )

    inference_spans = _spans_of_type(spans, SPAN_TYPES.INFERENCE)
    assert inference_spans, f"[{source}] expected at least one inference span"
    for span in inference_spans:
        assert span.attributes.get("span.subtype"), (
            f"[{source}] inference span has no subtype; the raw-response wrapper "
            "is probably not being unwrapped"
        )
        assert _output_event(span).get("response"), (
            f"[{source}] inference span captured no response — data.output was empty"
        )

    assert _spans_of_type(spans, SPAN_TYPES.AGENTIC_TOOL_INVOCATION), (
        f"[{source}] expected the booking tool to be traced"
    )


@requires_foundry
def test_foundry_agent_trace_is_complete():
    """A Foundry-backed run must report Foundry as host and capture the reply."""
    exporter, instrumentor = _instrument("ms_foundry_trace_test")
    try:
        asyncio.run(build_foundry_agent().run("Book a flight from Bombay to goa."))
        spans = exporter.get_finished_spans()
    finally:
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()

    _assert_foundry_trace(spans, source="local")


@requires_foundry
@pytest.mark.skipif(
    not (os.getenv("OKAHU_API_KEY") and OKAHU_WORKFLOW and OKAHU_APP),
    reason="OKAHU_API_KEY, MONOCLE_OKAHU_TEST_WORKFLOW or MONOCLE_OKAHU_TEST_APP not set",
)
def test_foundry_traces_reach_okahu():
    """The Foundry trace must survive the round trip to Okahu with its content.

    Asserting on the retrieved spans rather than the local ones is the point: it
    proves the attributes this patch fixes are still there after export and
    ingestion, not merely that some trace arrived.
    """
    from monocle_test_tools.okahu_span_loader import OkahuSpanLoader

    exporter, instrumentor = _instrument(OKAHU_WORKFLOW, with_okahu=True)
    try:
        asyncio.run(build_foundry_agent().run("Book a flight from Bombay to goa."))
        spans = exporter.get_finished_spans()
        assert spans, "the run produced no spans"
        # Every span of one run shares the trace id; take it from the workflow
        # span so the choice does not depend on export order.
        workflow_spans = _spans_of_type(spans, "workflow")
        assert workflow_spans, "expected a workflow span"
        trace_id = format(workflow_spans[0].context.trace_id, "032x")

        # Okahu's exporter batches, so nothing has necessarily left the process.
        provider = otel_trace.get_tracer_provider()
        if hasattr(provider, "force_flush"):
            provider.force_flush()
    finally:
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()

    # Ingestion is asynchronous; poll rather than assuming immediate availability.
    # Only retrieval errors are retried — the assertions run once, after the loop,
    # so a content failure is reported as itself rather than as a timeout.
    deadline = time.time() + 150
    retrieved = None
    last_error = None
    while time.time() < deadline:
        try:
            retrieved = OkahuSpanLoader.get_spans(
                workflow_name=OKAHU_APP, trace_id=trace_id
            )
        except Exception as exc:
            last_error = exc
            retrieved = None
        if retrieved:
            break
        time.sleep(10)

    if not retrieved:
        pytest.fail(
            f"trace {trace_id} was not readable from Okahu app {OKAHU_APP!r} within "
            f"the timeout; last error: {last_error}"
        )

    _assert_foundry_trace(retrieved, source="Okahu")
