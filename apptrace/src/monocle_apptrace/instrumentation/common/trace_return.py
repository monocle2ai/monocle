import base64
import gzip
import hmac
import importlib
import json
import logging
import os
import uuid

from monocle_apptrace.exporters.base_exporter import serialize_span
from monocle_apptrace.instrumentation.common.constants import (
    MONOCLE_TRACE_RETURN_ENABLED_ENV,
    MONOCLE_TRACE_RETRIEVAL_CALLBACK_ENV,
    MONOCLE_TRACE_RETRIEVAL_DEFAULT_KEY_ENV,
    TRACE_RETURN_REQUEST_HEADER,
    TRACE_RETURN_VERSION,
)

logger = logging.getLogger(__name__)

_DELIMITER_PREFIX = "__MONOCLE_TRACES__"


def is_trace_return_enabled() -> bool:
    return os.environ.get(MONOCLE_TRACE_RETURN_ENABLED_ENV, "false").lower() == "true"


def _get_header_case_insensitive(headers: dict, name: str):
    if not headers:
        return None
    lname = name.lower()
    for key, value in headers.items():
        if str(key).lower() == lname:
            return value
    return None


def default_trace_retrieval_callback(headers: dict) -> bool:
    """Default authorization: the request's x-monocle-retrieve-traces header
    value must equal the server key MONOCLE_TRACE_RETRIEVAL_DEFAULT_KEY."""
    key = os.environ.get(MONOCLE_TRACE_RETRIEVAL_DEFAULT_KEY_ENV)
    if not key:
        return False
    value = _get_header_case_insensitive(headers, TRACE_RETURN_REQUEST_HEADER)
    if value is None:
        return False
    return hmac.compare_digest(str(value), str(key))


def _resolve_callback(spec: str):
    """Resolve a 'pkg.module:callable' spec to a callable, or None on failure."""
    if ":" not in spec:
        logger.warning("Invalid MONOCLE_TRACE_RETRIEVAL_CALLBACK spec (expected 'module:callable'): %s", spec)
        return None
    module_path, _, attr = spec.partition(":")
    try:
        module = importlib.import_module(module_path)
        candidate = getattr(module, attr)
    except Exception as e:
        logger.warning("Could not load trace-retrieval callback '%s': %s", spec, e)
        return None
    if not callable(candidate):
        logger.warning("Trace-retrieval callback '%s' is not callable", spec)
        return None
    return candidate


def is_trace_return_authorized(headers: dict) -> bool:
    """Per-request authorization gate for trace retrieval. Uses the callback
    configured via MONOCLE_TRACE_RETRIEVAL_CALLBACK, or the default callback.
    Any failure to load or run the callback denies (returns False)."""
    spec = os.environ.get(MONOCLE_TRACE_RETRIEVAL_CALLBACK_ENV)
    if spec:
        callback = _resolve_callback(spec)
        if callback is None:
            return False
    else:
        callback = default_trace_retrieval_callback
    try:
        return bool(callback(headers))
    except Exception as e:
        logger.warning("Trace-retrieval authorization callback raised: %s", e)
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
