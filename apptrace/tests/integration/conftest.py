"""Shared fixtures for integration tests.

Monocle telemetry is process-global and doesn't re-initialize if already setup.
The autouse fixture resets global state between modules to prevent leakage.
"""

import pytest
from opentelemetry import trace
from opentelemetry.util._once import Once


def _reset_global_telemetry():
    """Fully reset Monocle + OpenTelemetry global tracing state."""
    from monocle_apptrace.instrumentation.common.instrumentor import (
        get_monocle_instrumentor,
        set_monocle_instrumentor,
        set_monocle_setup_signature,
        set_monocle_span_processor,
        set_tracer_provider,
    )

    instrumentor = get_monocle_instrumentor()
    if instrumentor is not None:
        try:
            if instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.uninstrument()
        except Exception:
            pass

    set_monocle_instrumentor(None)
    set_monocle_setup_signature(None)
    set_monocle_span_processor(None)
    set_tracer_provider(None)

    # OpenTelemetry only allows set_tracer_provider() to succeed once per process; reset
    # the guard so the next module can install its own provider (and thus its service.name).
    trace._TRACER_PROVIDER = None
    trace._TRACER_PROVIDER_SET_ONCE = Once()


@pytest.fixture(scope="module", autouse=True)
def reset_monocle_telemetry():
    _reset_global_telemetry()
    yield
    _reset_global_telemetry()
