import os
from monocle_apptrace.instrumentation.common import trace_return as tr


def test_enabled_reads_env(monkeypatch):
    monkeypatch.delenv("MONOCLE_ENABLE_TRACE_RETURN", raising=False)
    assert tr.is_trace_return_enabled() is False
    monkeypatch.setenv("MONOCLE_ENABLE_TRACE_RETURN", "true")
    assert tr.is_trace_return_enabled() is True
    monkeypatch.setenv("MONOCLE_ENABLE_TRACE_RETURN", "TRUE")
    assert tr.is_trace_return_enabled() is True
    monkeypatch.setenv("MONOCLE_ENABLE_TRACE_RETURN", "false")
    assert tr.is_trace_return_enabled() is False


def test_header_value_roundtrip():
    delim = tr.make_delimiter()
    value = tr.build_response_header_value(delim)
    assert value.startswith("v1;")
    assert tr.parse_delimiter_from_header(value) == delim
    assert tr.parse_delimiter_from_header("garbage") is None


def test_split_body_and_trailer():
    delim = "__MONOCLE_TR__abcd"
    clean = b'{"answer": "hi"}'
    trailer = delim.encode("utf-8") + b"PAYLOAD"
    body = clean + trailer
    got_clean, got_payload = tr.split_body_and_trailer(body, delim)
    assert got_clean == clean
    assert got_payload == "PAYLOAD"
    # no delimiter present -> payload None, body unchanged
    got_clean2, got_payload2 = tr.split_body_and_trailer(clean, delim)
    assert got_clean2 == clean
    assert got_payload2 is None


def test_encode_decode_roundtrip():
    class FakeSpan:
        def to_json(self):
            return '{"name": "inference", "status": {"status_code": "OK"}}'
    delim = tr.make_delimiter()
    trailer = tr.build_trailer_bytes([FakeSpan(), FakeSpan()], delim)
    body = b'{"answer": "hi"}' + trailer
    clean, payload = tr.split_body_and_trailer(body, delim)
    assert clean == b'{"answer": "hi"}'
    decoded = tr.decode_payload(payload)
    import json as _json
    spans = _json.loads(decoded)
    assert len(spans) == 2
    assert spans[0]["name"] == "inference"


def test_delimiter_is_header_safe():
    d = tr.make_delimiter()
    assert "\n" not in d and "\r" not in d
    assert d == d.strip()
    assert tr.parse_delimiter_from_header(tr.build_response_header_value(d)) == d
