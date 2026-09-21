"""Timeout resolution for the Okahu HTTP calls.

Precedence: an explicit timeout argument, then OKAHU_API_TIMEOUT, then 120.
The default has to be None rather than a number, or a caller taking the default
would be indistinguishable from one asking for that number and would silently
override the environment.
"""
import pytest

from monocle_test_tools.okahu_span_loader import OkahuSpanLoader

DEFAULT_TIMEOUT = 120


@pytest.fixture(autouse=True)
def _okahu_env(monkeypatch):
    monkeypatch.setenv("OKAHU_API_KEY", "test-key")
    monkeypatch.setenv("OKAHU_API_ENDPOINT", "https://api.example")
    monkeypatch.delenv("OKAHU_API_TIMEOUT", raising=False)


@pytest.fixture(name="get")
def get_fixture(monkeypatch):
    """Capture the timeout that actually reaches requests.get."""
    seen = {}

    class _Response:
        status_code = 200
        text = ""

        def raise_for_status(self):
            return None

        def json(self):
            return []

    def fake_get(url, headers=None, params=None, timeout=None):
        seen["timeout"] = timeout
        return _Response()

    monkeypatch.setattr("monocle_test_tools.okahu_span_loader.requests.get", fake_get)
    return seen


class TestResolution:

    def test_default_is_120(self):
        assert OkahuSpanLoader._resolve_timeout(None) == DEFAULT_TIMEOUT

    def test_env_overrides_the_default(self, monkeypatch):
        monkeypatch.setenv("OKAHU_API_TIMEOUT", "45")

        assert OkahuSpanLoader._resolve_timeout(None) == 45

    def test_an_explicit_timeout_wins_over_env(self, monkeypatch):
        monkeypatch.setenv("OKAHU_API_TIMEOUT", "45")

        assert OkahuSpanLoader._resolve_timeout(15) == 15

    def test_an_explicit_timeout_equal_to_the_default_is_still_honoured(self, monkeypatch):
        """The point of a None default: 120 passed in is a choice, not a fallback."""
        monkeypatch.setenv("OKAHU_API_TIMEOUT", "45")

        assert OkahuSpanLoader._resolve_timeout(DEFAULT_TIMEOUT) == DEFAULT_TIMEOUT

    @pytest.mark.parametrize("bad", ["", "  ", "abc", "12.5", "-1", "0"])
    def test_an_unusable_env_value_falls_back_to_the_default(self, monkeypatch, bad):
        """A misconfigured variable must not stop span loading."""
        monkeypatch.setenv("OKAHU_API_TIMEOUT", bad)

        assert OkahuSpanLoader._resolve_timeout(None) == DEFAULT_TIMEOUT


class TestItReachesTheRequest:

    def test_default_reaches_requests_get(self, get):
        OkahuSpanLoader.get_trace_ids("wf")

        assert get["timeout"] == DEFAULT_TIMEOUT

    def test_env_reaches_requests_get(self, get, monkeypatch):
        monkeypatch.setenv("OKAHU_API_TIMEOUT", "7")

        OkahuSpanLoader.get_trace_ids("wf")

        assert get["timeout"] == 7

    def test_an_explicit_argument_reaches_requests_get(self, get, monkeypatch):
        monkeypatch.setenv("OKAHU_API_TIMEOUT", "7")

        OkahuSpanLoader.get_trace_ids("wf", timeout=3)

        assert get["timeout"] == 3

    @pytest.mark.parametrize("call", [
        lambda: OkahuSpanLoader.get_trace_ids("wf"),
        lambda: OkahuSpanLoader.get_fact_ids("wf", "agent_requests"),
    ])
    def test_every_entry_point_picks_up_the_default(self, get, call):
        call()

        assert get["timeout"] == DEFAULT_TIMEOUT

    def test_no_signature_still_hardcodes_a_number(self):
        """A leftover numeric default would silently outrank the environment."""
        import inspect

        offenders = []
        for name in ("_get_resource", "_do_get", "get_trace_ids", "get_spans",
                     "get_fact_ids", "load_by_scope"):
            param = inspect.signature(getattr(OkahuSpanLoader, name)).parameters.get("timeout")
            if param is not None and param.default is not None:
                offenders.append(f"{name}={param.default!r}")

        assert offenders == []
