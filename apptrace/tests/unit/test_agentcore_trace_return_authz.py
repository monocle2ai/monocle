"""The deployed agent returns its spans only to a caller that presents the key.

AgentCore forwards a caller's headers to the agent only under its
X-Amzn-Bedrock-AgentCore-Runtime-Custom- prefix, so the handler strips that
before the shared is_trace_return_authorized decides. These cover that wiring;
the key-matching matrix itself belongs to test_trace_return_authz.py.
"""
import pytest

from monocle_apptrace.instrumentation.common import trace_return as tr
from monocle_apptrace.instrumentation.common.constants import (
    AGENTCORE_CUSTOM_HEADER_PREFIX,
    MONOCLE_TRACE_RETRIEVAL_CALLBACK_ENV,
    MONOCLE_TRACE_RETRIEVAL_DEFAULT_KEY_ENV,
    TRACE_RETURN_REQUEST_HEADER,
)
from monocle_apptrace.instrumentation.common.utils import remove_scopes
from monocle_apptrace.instrumentation.metamodel.agentcore.agentcore_handler import (
    AgentCoreSpanHandler,
)

KEY = "s3cret"
FORWARDED_HEADER = AGENTCORE_CUSTOM_HEADER_PREFIX + TRACE_RETURN_REQUEST_HEADER


class FakeRequest:
    """Stands in for the Starlette request _handle_invocation receives."""

    def __init__(self, headers=None):
        self.headers = headers if headers is not None else {}


@pytest.fixture
def deployment_expects_key(monkeypatch):
    monkeypatch.setattr(tr, "is_trace_return_enabled", lambda: True)
    monkeypatch.setenv(MONOCLE_TRACE_RETRIEVAL_DEFAULT_KEY_ENV, KEY)
    monkeypatch.delenv(MONOCLE_TRACE_RETRIEVAL_CALLBACK_ENV, raising=False)


def _authorizes(request) -> bool:
    """Whether the handler tagged this invocation for trace return.

    The scope is detached again straight away: pre_tracing attaches it to the
    ambient context, and a leaked one would make every later test in the process
    look like it had been authorized.
    """
    token, _ = AgentCoreSpanHandler().pre_tracing({}, None, None, (request,), {})
    if token is None:
        return False
    remove_scopes(token)
    return True


@pytest.mark.parametrize("headers, authorized", [
    ({FORWARDED_HEADER: KEY}, True),
    ({FORWARDED_HEADER.upper(): KEY}, True),
    ({TRACE_RETURN_REQUEST_HEADER: KEY}, True),
    ({FORWARDED_HEADER: "not-the-key"}, False),
    ({"content-type": "application/json"}, False),
], ids=["as agentcore forwards it", "however it is cased",
        "unprefixed, straight to the container", "wrong key", "ordinary caller"])
def test_the_key_is_found_however_it_arrives(deployment_expects_key, headers, authorized):
    assert _authorizes(FakeRequest(headers)) is authorized


@pytest.mark.parametrize("order", [0, 1], ids=["forwarded first", "bare first"])
def test_forwarded_header_wins_over_a_bare_one(deployment_expects_key, order):
    """A valid key under the forwarded name must not be undone by a stray one.

    Both names collapse to the same lookup, so whichever is applied last would
    otherwise decide — and the caller has no say in header order.
    """
    pair = [(FORWARDED_HEADER, KEY), (TRACE_RETURN_REQUEST_HEADER, "not-the-key")]

    assert _authorizes(FakeRequest(dict(pair if order == 0 else reversed(pair))))


def test_disabled_deployment_never_checks_the_key(monkeypatch):
    monkeypatch.setattr(tr, "is_trace_return_enabled", lambda: False)

    assert not _authorizes(FakeRequest({FORWARDED_HEADER: KEY}))


def test_unreadable_request_denies(deployment_expects_key):
    """Anything that is not a request the handler understands must not authorize."""
    assert not _authorizes(None)
    assert not _authorizes(FakeRequest(headers=None))


def test_custom_callback_sees_the_header_as_sent(monkeypatch):
    """A deployment's own callback can key off the real AgentCore header name."""
    monkeypatch.setattr(tr, "is_trace_return_enabled", lambda: True)
    monkeypatch.setenv(MONOCLE_TRACE_RETRIEVAL_CALLBACK_ENV,
                       f"{__name__}:only_the_forwarded_header")

    assert _authorizes(FakeRequest({FORWARDED_HEADER: KEY}))
    assert not _authorizes(FakeRequest({TRACE_RETURN_REQUEST_HEADER: KEY}))


def only_the_forwarded_header(headers: dict) -> bool:
    return headers.get(FORWARDED_HEADER) == KEY


if __name__ == "__main__":
    pytest.main([__file__])
