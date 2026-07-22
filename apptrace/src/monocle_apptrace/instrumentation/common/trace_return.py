import base64
import gzip
import json
import os
import uuid

from monocle_apptrace.exporters.base_exporter import serialize_span
from monocle_apptrace.instrumentation.common.constants import (
    MONOCLE_TRACE_RETURN_ENABLED_ENV,
    TRACE_RETURN_REQUEST_HEADER,
    TRACE_RETURN_VERSION,
)

_DELIMITER_PREFIX = "__MONOCLE_TRACES__"


def is_trace_return_enabled() -> bool:
    return os.environ.get(MONOCLE_TRACE_RETURN_ENABLED_ENV, "false").lower() == "true"


def is_trace_return_requested(headers: dict) -> bool:
    if not headers:
        return False
    for key, value in headers.items():
        if str(key).lower() == TRACE_RETURN_REQUEST_HEADER and str(value).lower() == "true":
            return True
    return False


def make_delimiter() -> str:
    return f"{_DELIMITER_PREFIX}{uuid.uuid4().hex}__"


def encode_spans(spans: list) -> str:
    span_dicts = [serialize_span(span) for span in spans]
    raw = json.dumps(span_dicts).encode("utf-8")
    return base64.b64encode(gzip.compress(raw)).decode("ascii")


def decode_payload(payload: str) -> str:
    raw = gzip.decompress(base64.b64decode(payload.encode("ascii")))
    return raw.decode("utf-8")


def build_trailer_bytes(spans: list, delimiter: str) -> bytes:
    return delimiter.encode("utf-8") + encode_spans(spans).encode("ascii")


def build_response_header_value(delimiter: str) -> str:
    return f"{TRACE_RETURN_VERSION}; delim={delimiter}"


def parse_delimiter_from_header(header_value: str) -> "str | None":
    if not header_value or "delim=" not in header_value:
        return None
    return header_value.split("delim=", 1)[1].strip()


def split_body_and_trailer(body: bytes, delimiter: str) -> "tuple[bytes, str | None]":
    marker = delimiter.encode("utf-8")
    idx = body.find(marker)
    if idx == -1:
        return body, None
    clean = body[:idx]
    payload = body[idx + len(marker):].decode("ascii")
    return clean, payload
