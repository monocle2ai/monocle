"""Token usage from Microsoft Copilot's native OTel JSONL export.

Copilot doesn't surface token counts through hooks, but its OTel file export does —
as metadata, so no `captureContent` (hence no content-policy permission) is needed.
Two surfaces write to that file in different OTel signal shapes:

  - VS Code Copilot Chat → an inference *log event*
    (`gen_ai.client.inference.operation.details`); trace id under `spanContext`,
    time in `hrTime`.
  - Copilot CLI → a *chat span* (`type: "span"`, `gen_ai.operation.name: "chat"`);
    trace id at top level, time in `startTime`.

Both carry identical `gen_ai.usage.*` attributes. `_normalize_record` is the single
point that collapses either shape into one canonical record; everything downstream
works on that normalized stream and is surface-agnostic.

Correlation is anchored on the **trace id**: find the inference records whose
timestamp falls in the turn window, take the dominant trace id, then sum every
record sharing it. Utility calls (title/summary) carry no trace id and drop out.
"""

import json
import logging
import os
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_INFERENCE_EVENT = "gen_ai.client.inference.operation.details"
_WINDOW_SLACK_S = 5.0  # tolerate clock/flush skew between hook timestamps and OTel hrTime
# Copilot's OTel exporter flushes the file asynchronously, so the turn's inference
# event can land a beat after the Stop hook fires. Poll briefly for it to appear.
_FLUSH_WAIT_S = 6.0
_FLUSH_POLL_INTERVAL_S = 0.5

# Keep ~1 day of records, pruned on SessionStart, so the append-only OTel file
# can't grow unbounded (matches trace_events._TTL_SECONDS).
_OTEL_TTL_SECONDS = 24 * 60 * 60


def _otel_file() -> Path:
    """Where both Copilot surfaces write their OTel export — a FIXED root path,
    ~/.monocle/.copilot_otel/copilot.jsonl. VS Code only reliably file-exports to a
    stable outfile, so this must not move per-project. The CLI passes the path via
    env (wins when present); the recorded env-file value is a secondary fallback."""
    env = os.environ.get("COPILOT_OTEL_FILE_EXPORTER_PATH")
    if env:
        return Path(env)
    from monocle_apptrace.instrumentation.common.utils import get_monocle_env_value
    configured = get_monocle_env_value("MONOCLE_COPILOT_OTEL_FILE")
    if configured:
        return Path(configured)
    return Path.home() / ".monocle" / ".copilot_otel" / "copilot.jsonl"


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


def _normalize_record(rec: dict):
    """The single normalization point: map either Copilot OTel envelope — VS Code
    Chat's inference *log event* or the CLI's *chat span* — into one canonical token
    record, or None if the record isn't an inference. Both envelopes carry the same
    `gen_ai.usage.*` attributes; only the trace-id and timestamp locations differ.
    """
    if not isinstance(rec, dict):
        return None
    attrs = rec.get("attributes") or {}
    is_chat_log = attrs.get("event.name") == _INFERENCE_EVENT                        # VS Code Copilot Chat
    is_chat_span = rec.get("type") == "span" and attrs.get("gen_ai.operation.name") == "chat"  # Copilot CLI
    if not (is_chat_log or is_chat_span):
        return None
    return {
        "trace_id": rec.get("traceId") or (rec.get("spanContext") or {}).get("traceId") or "",
        "epoch": _hrtime_to_epoch(rec.get("startTime") or rec.get("hrTime")),
        "response_id": attrs.get("gen_ai.response.id") or "",
        "input": attrs.get("gen_ai.usage.input_tokens", 0) or 0,
        "output": attrs.get("gen_ai.usage.output_tokens", 0) or 0,
        "cache_read": attrs.get("gen_ai.usage.cache_read.input_tokens", 0) or 0,
        "reasoning": attrs.get("gen_ai.usage.reasoning.output_tokens", 0) or 0,
        "model": attrs.get("gen_ai.response.model") or attrs.get("gen_ai.request.model") or "",
    }


def _load_inference_events(path: Path) -> list:
    """Read the OTel JSONL and return the normalized inference records (both surfaces)."""
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as e:
        logger.debug("Cannot read Copilot OTel file %s: %s", path, e)
        return []
    events = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        norm = _normalize_record(rec)
        if norm is not None:
            events.append(norm)
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


def lookup_turn_tokens(turn_start: str, turn_end: str, wait_s: float = _FLUSH_WAIT_S):
    """Return (tokens_dict, trace_id, model) for the turn, or ({}, "", "").

    tokens_dict uses Monocle's canonical keys. trace_id is returned so the caller
    can stamp it on the span (provenance for the correlation).

    Polls the OTel file for up to wait_s seconds: Copilot's exporter flushes
    asynchronously, so at Stop-hook time the turn's inference event may not be on
    disk yet. We re-read until an event in the turn window appears (or we give up).
    """
    start = _iso_to_epoch(turn_start)
    end = _iso_to_epoch(turn_end)
    deadline = time.monotonic() + max(0.0, wait_s)
    events = []
    trace_id = ""
    while True:
        events = _load_inference_events(_otel_file())
        trace_id = _resolve_trace_id(events, start, end) if events else ""
        if trace_id or time.monotonic() >= deadline:
            break
        time.sleep(_FLUSH_POLL_INTERVAL_S)

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


def prune_otel_file(ttl_seconds: float = _OTEL_TTL_SECONDS) -> int:
    """Drop records older than ttl_seconds from the append-only OTel file so it can't
    grow unbounded. Pruned by record time, not mtime (the file is always being
    appended to). Called on SessionStart; returns the number of records dropped."""
    path = _otel_file()
    if not path.exists():
        return 0
    cutoff = time.time() - ttl_seconds
    kept, dropped = [], 0
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                kept.append(line)  # keep lines we can't parse rather than lose them
                continue
            epoch = _hrtime_to_epoch(rec.get("startTime") or rec.get("hrTime"))
            if epoch is not None and epoch < cutoff:
                dropped += 1
            else:
                kept.append(line)
        if dropped:
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
            tmp.replace(path)  # atomic swap; a reader never sees a partial file
    except OSError as e:
        logger.debug("prune_otel_file: %s", e)
    return dropped
