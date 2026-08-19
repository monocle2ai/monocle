"""Health check sampling in HttpSpanHandler.should_sample.

Health check endpoints answer with a body (eg {"status":"ok"}), so a non empty response
alone can't mark a span as real traffic. A failing health check must always be exported,
no matter how empty its output is.
"""
import pytest
from opentelemetry.trace.status import Status, StatusCode

from monocle_apptrace.instrumentation.common.constants import HEALTH_RESET_COUNTER
from monocle_apptrace.instrumentation.common.span_handler import HttpSpanHandler, http_span_counter


class FakeEvent:
    def __init__(self, name, attributes):
        self.name = name
        self.attributes = attributes


class FakeSpan:
    def __init__(self, method="GET", route="/healthz", url=None, response=None, params=None,
                 status_code="200", span_status=StatusCode.OK):
        self.attributes = {"entity.1.method": method, "entity.1.route": route,
                           "entity.1.url": url or f"http://10.1.2.3:8080{route}"}
        input_attributes = {"params": params} if params else {}
        output_attributes = {}
        if status_code is not None:
            output_attributes["status_code"] = status_code
        if response is not None:
            output_attributes["response"] = response
        self.events = [FakeEvent("data.input", input_attributes),
                       FakeEvent("data.output", output_attributes)]
        self.status = Status(span_status)


@pytest.fixture(autouse=True)
def reset_counter():
    http_span_counter.reset()
    yield
    http_span_counter.reset()


def should_sample(span):
    return HttpSpanHandler().should_sample(None, None, None, None, None, None, None, span, None)


def sample_count(span_factory, requests=HEALTH_RESET_COUNTER):
    """Exports seen over `requests` calls. Health checks export 1 per HEALTH_RESET_COUNTER."""
    return sum(1 for _ in range(requests) if should_sample(span_factory()))


@pytest.mark.parametrize("route", ["/healthz", "/health", "/livez", "/readyz", "/ping",
                                   "/actuator/health", "/api/v1/healthz", "/healthz/"])
def test_health_check_with_response_body_is_sampled_out(route):
    """Health checks that return a payload used to be exported on every single request."""
    assert sample_count(lambda: FakeSpan(route=route, response='{"status": "ok"}')) == 1


def test_health_check_with_plain_text_body_is_sampled_out():
    assert sample_count(lambda: FakeSpan(route="/livez", response="OK")) == 1


def test_health_check_with_empty_body_is_sampled_out():
    assert sample_count(lambda: FakeSpan(route="/healthz")) == 1


@pytest.mark.parametrize("status_code, span_status", [("503", StatusCode.ERROR),
                                                      ("500", StatusCode.OK),
                                                      ("404", StatusCode.OK)])
def test_failing_health_check_is_always_exported(status_code, span_status):
    """A failing health check is the one health check span you must not lose."""
    assert sample_count(lambda: FakeSpan(status_code=status_code, span_status=span_status), requests=3) == 3


@pytest.mark.parametrize("status_code", ["203", "304"])
def test_health_check_answering_2xx_or_3xx_is_sampled_out(status_code):
    """Anything below 400 counts as a successful health check."""
    assert sample_count(lambda: FakeSpan(status_code=status_code)) == 1


def test_failing_health_check_reported_as_error_code_is_always_exported():
    """lambda/agentcore metamodels report the status under error_code instead."""
    def span_factory():
        span = FakeSpan(status_code=None)
        span.events[1].attributes["error_code"] = "500"
        return span
    assert sample_count(span_factory, requests=3) == 3


def test_regular_route_with_response_body_is_always_exported():
    assert sample_count(lambda: FakeSpan(route="/hello", response='{"Status": "Success"}'), requests=3) == 3


def test_health_check_route_with_request_params_is_always_exported():
    """Query params mean a real caller, not a health check."""
    assert sample_count(lambda: FakeSpan(route="/healthz", params="verbose=1",
                                         response='{"status": "ok"}'), requests=3) == 3


def test_post_request_is_always_exported():
    assert sample_count(lambda: FakeSpan(method="POST", route="/healthz"), requests=3) == 3


def test_sampling_disabled_exports_every_health_check(monkeypatch):
    monkeypatch.setattr(HttpSpanHandler, "sample_health_checks", False)
    assert sample_count(lambda: FakeSpan(response='{"status": "ok"}'), requests=3) == 3


def test_custom_health_check_routes(monkeypatch):
    monkeypatch.setattr(HttpSpanHandler, "health_check_routes", ["/status-check"])
    assert sample_count(lambda: FakeSpan(route="/status-check", response="OK")) == 1
    http_span_counter.reset()
    assert sample_count(lambda: FakeSpan(route="/healthz", response="OK"), requests=3) == 3


def test_exception_is_always_exported():
    handler = HttpSpanHandler()
    span = FakeSpan()
    assert all(handler.should_sample(None, None, None, None, None, None, ValueError("boom"), span, None)
               for _ in range(3))
