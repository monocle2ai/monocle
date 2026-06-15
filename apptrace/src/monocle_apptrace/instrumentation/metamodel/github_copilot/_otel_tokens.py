"""Token usage from GitHub Copilot's native OTel JSONL export."""

import json
import logging
import os
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_INFERENCE_EVENT = "gen_ai.client.inference.operation.details"
_WINDOW_SLACK_S = 5.0
_METRIC_FLUSH_SLACK_S = 15.0


def _otel_file() -> Path:
    env = os.environ.get("COPILOT_OTEL_FILE_EXPORTER_PATH")
    if env:
        return Path(env)
    from monocle_apptrace.instrumentation.common.utils import get_monocle_env_value
    return Path(get_monocle_env_value("MONOCLE_COPILOT_OTEL_FILE") or Path.home() / ".monocle" / ".copilot_otel" / "copilot.jsonl")

def _iso_to_epoch(ts):
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp() if ts else None
    except ValueError:
        return None

def _hrtime_to_epoch(value):
    try:
        return float(value[0]) + float(value[1]) / 1e9 if isinstance(value, list) and len(value) == 2 else None
    except (TypeError, ValueError):
        return None

def _as_int(value):
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0

def _tokens(input_tokens, output_tokens, trace_id="", model="", cache=0, reasoning=0):
    if not input_tokens and not output_tokens:
        return None
    result = {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    if cache:
        result["cache_read_tokens"] = cache
    if reasoning:
        result["reasoning_tokens"] = reasoning
    return result, trace_id, model

def _load_records(path):
    direct, metrics = [], []
    if not path.exists():
        return direct, metrics
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as e:
        logger.debug("Cannot read Copilot OTel file %s: %s", path, e)
        return direct, metrics

    for line in lines:
        if "gen_ai.usage" not in line and "gen_ai.client.token.usage" not in line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue

        attrs = rec.get("attributes") or {}
        if attrs.get("event.name") == _INFERENCE_EVENT or (rec.get("type") == "span" and attrs.get("gen_ai.operation.name") == "chat"):
            direct.append({
                "epoch": _hrtime_to_epoch(rec.get("startTime") or rec.get("hrTime")),
                "trace_id": rec.get("traceId") or (rec.get("spanContext") or {}).get("traceId") or "",
                "response_id": attrs.get("gen_ai.response.id") or "",
                "model": attrs.get("gen_ai.response.model") or attrs.get("gen_ai.request.model") or "",
                "input": _as_int(attrs.get("gen_ai.usage.input_tokens")),
                "output": _as_int(attrs.get("gen_ai.usage.output_tokens")),
                "cache": _as_int(attrs.get("gen_ai.usage.cache_read.input_tokens")),
                "reasoning": _as_int(attrs.get("gen_ai.usage.reasoning.output_tokens")),
            })

        try:
            session_id = dict((rec.get("resource") or {}).get("_rawAttributes") or []).get("session.id", "")
        except (TypeError, ValueError):
            session_id = ""
        grouped = {}
        for scope in rec.get("scopeMetrics") or []:
            scope_name = (scope.get("scope") or {}).get("name", "")
            for metric in scope.get("metrics") or []:
                if (metric.get("descriptor") or {}).get("name") != "gen_ai.client.token.usage":
                    continue
                for point in metric.get("dataPoints") or []:
                    attrs = point.get("attributes") or {}
                    token_type = attrs.get("gen_ai.token.type")
                    epoch = _hrtime_to_epoch(point.get("endTime"))
                    if attrs.get("gen_ai.operation.name") != "chat" or token_type not in ("input", "output") or epoch is None:
                        continue
                    model = attrs.get("gen_ai.response.model") or attrs.get("gen_ai.request.model") or ""
                    key = (session_id, scope_name, attrs.get("gen_ai.provider.name", ""), model)
                    snap = grouped.setdefault(key, {
                        "epoch": epoch, "start_epoch": _hrtime_to_epoch(point.get("startTime")),
                        "session_id": session_id, "scope": scope_name, "model": model,
                        "input_sum": 0, "input_count": 0, "output_sum": 0, "output_count": 0,
                    })
                    snap["epoch"] = max(snap["epoch"], epoch)
                    start_epoch = _hrtime_to_epoch(point.get("startTime"))
                    starts = [v for v in (snap["start_epoch"], start_epoch) if v is not None]
                    snap["start_epoch"] = min(starts) if starts else None
                    value = point.get("value") or {}
                    snap[f"{token_type}_sum"] = _as_int(value.get("sum"))
                    snap[f"{token_type}_count"] = _as_int(value.get("count"))
        metrics.extend(grouped.values())
    return direct, metrics

def _direct_tokens(records, start, end, expected_model=""):
    in_window = [
        r for r in records
        if r["epoch"] is not None
        and (start is None or r["epoch"] >= start - _WINDOW_SLACK_S)
        and (end is None or r["epoch"] <= end + _WINDOW_SLACK_S)
        and (not expected_model or not r["model"] or r["model"] == expected_model)
    ]
    if not in_window:
        return None

    trace_ids = [r["trace_id"] for r in in_window if r["trace_id"]]
    trace_id = Counter(trace_ids).most_common(1)[0][0] if trace_ids else ""
    seen, deduped = set(), []
    for record in ([r for r in records if r["trace_id"] == trace_id] if trace_id else in_window):
        key = (record["response_id"], record["input"], record["output"])
        if key not in seen:
            seen.add(key)
            deduped.append(record)

    models = Counter(r["model"] for r in deduped if r["model"])
    return _tokens(
        sum(r["input"] for r in deduped),
        sum(r["output"] for r in deduped),
        trace_id,
        models.most_common(1)[0][0] if models else "",
        sum(r["cache"] for r in deduped),
        sum(r["reasoning"] for r in deduped),
    )


def _metric_tokens(snapshots, start, end, expected_model=""):
    latest_before, first_after = {}, {}
    for snap in snapshots:
        if expected_model and snap["model"] and snap["model"] != expected_model:
            continue
        key = (snap["session_id"], snap["scope"], snap["model"])
        if start is None or snap["epoch"] <= start:
            if key not in latest_before or snap["epoch"] > latest_before[key]["epoch"]:
                latest_before[key] = snap
        if end is None or end <= snap["epoch"] <= end + _METRIC_FLUSH_SLACK_S:
            if key not in first_after or snap["epoch"] < first_after[key]["epoch"]:
                first_after[key] = snap

    candidates = []
    for key, after in first_after.items():
        before = latest_before.get(key)
        if before is None:
            if after["start_epoch"] is None or start is None or after["start_epoch"] < start - _WINDOW_SLACK_S:
                continue
            before = {"input_sum": 0, "input_count": 0, "output_sum": 0, "output_count": 0}
        if after["input_count"] <= before["input_count"] and after["output_count"] <= before["output_count"]:
            continue
        result = _tokens(
            max(0, after["input_sum"] - before["input_sum"]),
            max(0, after["output_sum"] - before["output_sum"]),
            model=after["model"],
        )
        if result:
            candidates.append(result)
    return max(candidates, key=lambda item: item[0]["total_tokens"]) if candidates else None


def lookup_turn_tokens(turn_start, turn_end, wait_s=6.0, expected_model=""):
    start, end = _iso_to_epoch(turn_start), _iso_to_epoch(turn_end)
    deadline = time.monotonic() + max(0.0, wait_s)
    while True:
        direct, metrics = _load_records(_otel_file())
        result = _direct_tokens(direct, start, end, expected_model) or _metric_tokens(metrics, start, end, expected_model)
        if result or time.monotonic() >= deadline:
            return result or ({}, "", "")
        time.sleep(0.5)


def prune_otel_file(ttl_seconds=10 * 60):
    path = _otel_file()
    if not path.exists():
        return 0

    cutoff, kept, dropped = time.time() - ttl_seconds, [], 0
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                kept.append(line)
                continue

            epoch = _hrtime_to_epoch(rec.get("startTime") or rec.get("hrTime"))
            if epoch is None:
                epochs = [
                    _hrtime_to_epoch(point.get("endTime"))
                    for scope in rec.get("scopeMetrics") or []
                    for metric in scope.get("metrics") or []
                    if (metric.get("descriptor") or {}).get("name") == "gen_ai.client.token.usage"
                    for point in metric.get("dataPoints") or []
                ]
                epoch = max((e for e in epochs if e is not None), default=None)
            if epoch is not None and epoch < cutoff:
                dropped += 1
            else:
                kept.append(line)

        if dropped:
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
            tmp.replace(path)
    except OSError as e:
        logger.debug("prune_otel_file: %s", e)
    return dropped
