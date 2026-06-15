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

Some Copilot CLI builds only export token usage as OTel histogram metrics. Those
records have no trace id, but the histogram's `count` and `sum` let us recover
new chat observations that landed during a turn window. That path is a time-window
fallback, not trace-correlated provenance.
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
_METRIC_FLUSH_SLACK_S = 15.0
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


def _load_token_metric_snapshots(path: Path) -> list:
    """Read cumulative token-usage histogram snapshots from Copilot's OTel export."""
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as e:
        logger.debug("Cannot read Copilot OTel file %s: %s", path, e)
        return []

    snapshots = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        scope_metrics = rec.get("scopeMetrics")
        if not isinstance(scope_metrics, list):
            continue
        resource_attrs = dict((rec.get("resource") or {}).get("_rawAttributes") or [])
        session_id = resource_attrs.get("session.id", "")
        for scope in scope_metrics:
            scope_name = (scope.get("scope") or {}).get("name", "")
            grouped = {}
            for metric in scope.get("metrics") or []:
                if (metric.get("descriptor") or {}).get("name") != "gen_ai.client.token.usage":
                    continue
                for point in metric.get("dataPoints") or []:
                    attrs = point.get("attributes") or {}
                    if attrs.get("gen_ai.operation.name") != "chat":
                        continue
                    token_type = attrs.get("gen_ai.token.type")
                    if token_type not in ("input", "output"):
                        continue
                    epoch = _hrtime_to_epoch(point.get("endTime"))
                    point_start = _hrtime_to_epoch(point.get("startTime"))
                    if epoch is None:
                        continue
                    value = point.get("value") or {}
                    try:
                        token_sum = int(value.get("sum", 0) or 0)
                        token_count = int(value.get("count", 0) or 0)
                    except (TypeError, ValueError):
                        continue
                    model = attrs.get("gen_ai.response.model") or attrs.get("gen_ai.request.model") or ""
                    key = (
                        session_id,
                        scope_name,
                        attrs.get("gen_ai.provider.name", ""),
                        attrs.get("gen_ai.operation.name", ""),
                        model,
                    )
                    item = grouped.setdefault(key, {
                        "epoch": epoch,
                        "start_epoch": point_start,
                        "session_id": session_id,
                        "scope": scope_name,
                        "model": model,
                        "input_sum": 0,
                        "input_count": 0,
                        "output_sum": 0,
                        "output_count": 0,
                    })
                    item["epoch"] = max(item["epoch"], epoch)
                    starts = [v for v in (item.get("start_epoch"), point_start) if v is not None]
                    item["start_epoch"] = min(starts) if starts else None
                    item[f"{token_type}_sum"] = token_sum
                    item[f"{token_type}_count"] = token_count
            snapshots.extend(grouped.values())
    return snapshots


def _model_matches(model: str, expected_model: str = "") -> bool:
    return not expected_model or not model or model == expected_model


def _metric_histogram_delta_tokens(path: Path, start, end, expected_model: str = ""):
    snapshots = _load_token_metric_snapshots(path)
    if not snapshots:
        return {}, ""

    latest_before = {}
    first_after = {}
    for snap in snapshots:
        if not _model_matches(snap.get("model", ""), expected_model):
            continue
        key = (snap["session_id"], snap["scope"], snap["model"])
        epoch = snap["epoch"]
        if start is None or epoch <= start:
            prev = latest_before.get(key)
            if prev is None or epoch > prev["epoch"]:
                latest_before[key] = snap
        if end is None or end <= epoch <= end + _METRIC_FLUSH_SLACK_S:
            nxt = first_after.get(key)
            if nxt is None or epoch < nxt["epoch"]:
                first_after[key] = snap

    best = None
    for key, after in first_after.items():
        before = latest_before.get(key)
        if before is None:
            point_start = after.get("start_epoch")
            if point_start is None or start is None or point_start < start - _WINDOW_SLACK_S:
                continue
            before = {
                "input_sum": 0,
                "input_count": 0,
                "output_sum": 0,
                "output_count": 0,
            }
        input_count = after["input_count"] - before.get("input_count", 0)
        output_count = after["output_count"] - before.get("output_count", 0)
        if input_count <= 0 and output_count <= 0:
            continue
        inp = max(0, after["input_sum"] - before.get("input_sum", 0))
        out = max(0, after["output_sum"] - before.get("output_sum", 0))
        if not inp and not out:
            continue
        candidate = {"input": inp, "output": out, "model": after.get("model", "")}
        if best is None or (inp + out) > (best["input"] + best["output"]):
            best = candidate

    if not best:
        return {}, ""
    return {
        "prompt_tokens": best["input"],
        "completion_tokens": best["output"],
        "total_tokens": best["input"] + best["output"],
    }, best.get("model", "")


def _resolve_trace_id(events: list, start, end, expected_model: str = "") -> str:
    """The dominant trace id among events whose timestamp is inside the turn window."""
    in_window = _events_in_window(events, start, end, expected_model=expected_model)
    with_trace = [e for e in in_window if e["trace_id"]]
    if not with_trace:
        return ""
    return Counter(e["trace_id"] for e in with_trace).most_common(1)[0][0]


def _events_in_window(events: list, start, end, expected_model: str = "") -> list:
    """Inference records whose timestamp falls inside the turn window."""
    in_window = [
        e for e in events
        if e["epoch"] is not None
        and _model_matches(e.get("model", ""), expected_model)
        and (start is None or e["epoch"] >= start - _WINDOW_SLACK_S)
        and (end is None or e["epoch"] <= end + _WINDOW_SLACK_S)
    ]
    return in_window


def lookup_turn_tokens(
    turn_start: str,
    turn_end: str,
    wait_s: float = _FLUSH_WAIT_S,
    expected_model: str = "",
):
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
    path = _otel_file()
    events = []
    trace_id = ""
    while True:
        events = _load_inference_events(path)
        trace_id = _resolve_trace_id(events, start, end, expected_model=expected_model) if events else ""
        if (
            trace_id
            or _events_in_window(events, start, end, expected_model=expected_model)
            or time.monotonic() >= deadline
        ):
            break
        time.sleep(_FLUSH_POLL_INTERVAL_S)

    if trace_id:
        group = [e for e in events if e["trace_id"] == trace_id]
    else:
        # Some Copilot OTel log records carry token usage but no trace id. Fall
        # back to the turn time window so tokens are not dropped entirely.
        group = _events_in_window(events, start, end, expected_model=expected_model)

    if not group:
        tokens, model = _metric_histogram_delta_tokens(path, start, end, expected_model=expected_model)
        return tokens, "", model
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
