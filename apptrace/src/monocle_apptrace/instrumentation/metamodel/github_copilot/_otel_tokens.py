"""Token usage from Microsoft Copilot's native OTel JSONL export.

VS Code Copilot Chat does not surface token counts through hooks, but its OTel
file export does — as metadata, so no `captureContent` (and thus no content-policy
permission) is needed. Each LLM call is a `gen_ai.client.inference.operation.details`
event carrying `gen_ai.usage.*` and a `spanContext.traceId`.

Correlation is anchored on the **trace id**: we find the inference events whose
timestamp falls in the turn's window, take the dominant trace id there, then sum
every event sharing that trace id. Utility calls (title/summary) carry no trace id
and are naturally excluded.
"""

import json
import logging
import os
from collections import Counter
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_INFERENCE_EVENT = "gen_ai.client.inference.operation.details"
_WINDOW_SLACK_S = 5.0  # tolerate clock/flush skew between hook timestamps and OTel hrTime


def _otel_file() -> Path:
    """Same file the setup points both Copilot surfaces at."""
    env = os.environ.get("COPILOT_OTEL_FILE_EXPORTER_PATH")
    return Path(env) if env else Path.home() / ".monocle" / ".copilot_otel" / "copilot.jsonl"


def _iso_to_epoch(ts: str):
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _hrtime_to_epoch(hrtime):
    """OTel SDK records time as [seconds, nanoseconds]."""
    if not isinstance(hrtime, list) or len(hrtime) != 2:
        return None
    try:
        return float(hrtime[0]) + float(hrtime[1]) / 1e9
    except (TypeError, ValueError):
        return None


def _load_inference_events(path: Path) -> list:
    if not path.exists():
        return []
    events = []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as e:
        logger.debug("Cannot read Copilot OTel file %s: %s", path, e)
        return []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(rec, dict):
            continue
        attrs = rec.get("attributes") or {}
        if attrs.get("event.name") != _INFERENCE_EVENT:
            continue
        events.append({
            "trace_id": (rec.get("spanContext") or {}).get("traceId") or "",
            "epoch": _hrtime_to_epoch(rec.get("hrTime")),
            "response_id": attrs.get("gen_ai.response.id") or "",
            "input": attrs.get("gen_ai.usage.input_tokens", 0) or 0,
            "output": attrs.get("gen_ai.usage.output_tokens", 0) or 0,
            "cache_read": attrs.get("gen_ai.usage.cache_read.input_tokens", 0) or 0,
            "reasoning": attrs.get("gen_ai.usage.reasoning.output_tokens", 0) or 0,
            "model": attrs.get("gen_ai.response.model") or attrs.get("gen_ai.request.model") or "",
        })
    return events


def _resolve_trace_id(events: list, start, end) -> str:
    """The dominant trace id among events whose timestamp is inside the turn window."""
    in_window = [
        e for e in events
        if e["trace_id"] and e["epoch"] is not None
        and (start is None or e["epoch"] >= start - _WINDOW_SLACK_S)
        and (end is None or e["epoch"] <= end + _WINDOW_SLACK_S)
    ]
    if not in_window:
        return ""
    return Counter(e["trace_id"] for e in in_window).most_common(1)[0][0]


def lookup_turn_tokens(turn_start: str, turn_end: str):
    """Return (tokens_dict, trace_id, model) for the turn, or ({}, "", "").

    tokens_dict uses Monocle's canonical keys. trace_id is returned so the caller
    can stamp it on the span (provenance for the correlation).
    """
    events = _load_inference_events(_otel_file())
    if not events:
        return {}, "", ""

    trace_id = _resolve_trace_id(events, _iso_to_epoch(turn_start), _iso_to_epoch(turn_end))
    if not trace_id:
        return {}, "", ""

    group = [e for e in events if e["trace_id"] == trace_id]
    # Drop byte-identical duplicate emissions of the same call (same response id
    # AND same counts); keep distinct rounds even when a response id repeats.
    seen = set()
    deduped = []
    for e in group:
        key = (e["response_id"], e["input"], e["output"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(e)

    inp = sum(e["input"] for e in deduped)
    out = sum(e["output"] for e in deduped)
    cache = sum(e["cache_read"] for e in deduped)
    reasoning = sum(e["reasoning"] for e in deduped)
    if not inp and not out:
        return {}, trace_id, ""

    tokens = {"prompt_tokens": inp, "completion_tokens": out, "total_tokens": inp + out}
    if cache:
        tokens["cache_read_tokens"] = cache
    if reasoning:
        tokens["reasoning_tokens"] = reasoning

    models = Counter(e["model"] for e in deduped if e["model"])
    model = models.most_common(1)[0][0] if models else ""
    return tokens, trace_id, model
