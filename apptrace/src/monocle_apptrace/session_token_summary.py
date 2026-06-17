"""Per-session token usage summary from local Monocle trace files.

Reads .monocle/monocle_trace_*.json files and aggregates token counts
per session and model, displayed as a table in the terminal.
"""

import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

MONOCLE_DIR = Path.home() / ".monocle"

SESSION_ATTR = "scope.agentic.session"

_TOKEN_KEYS = [
    "prompt_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
    "completion_tokens",
    "total_tokens",
]

_HEADERS = ["Session", "Model", "Input", "Cache Read", "Cache Create", "Output", "Total"]


def _parse_timestamp_from_filename(name):
    # type: (str) -> Optional[datetime]
    if name.endswith(".json"):
        stem = name[:-5]
    else:
        stem = name
    parts = stem.rsplit("_", 2)
    if len(parts) < 3:
        return None
    try:
        return datetime.strptime(
            "{}_{}".format(parts[-2], parts[-1]), "%Y-%m-%d_%H.%M.%S"
        ).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _window_cutoff(time_window):
    # type: (str) -> Optional[datetime]
    now = datetime.now(tz=timezone.utc)
    w = time_window.lower().strip()
    if w == "today":
        return now.replace(hour=0, minute=0, second=0, microsecond=0)
    if w == "this week":
        return (now - timedelta(days=now.weekday())).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
    if "days" in w:
        try:
            n = int(w.split()[0])
            return now - timedelta(days=n)
        except (ValueError, IndexError):
            pass
    return None


def _load_spans(path):
    # type: (Path) -> List[Dict]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def summarize(time_window="all", monocle_dir=None):
    # type: (str, Optional[Path]) -> List[Dict]
    """Aggregate token usage per session from trace files in *monocle_dir*.

    Returns a list of row dicts sorted by session then model, each with keys:
    session, model, prompt_tokens, cache_read_input_tokens,
    cache_creation_input_tokens, completion_tokens, total_tokens.
    """
    if monocle_dir is None:
        monocle_dir = MONOCLE_DIR

    cutoff = _window_cutoff(time_window)

    # aggregated[session_id][model][token_key] = cumulative int
    aggregated: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    if not monocle_dir.exists():
        return []

    for trace_file in sorted(monocle_dir.glob("monocle_trace_*.json")):
        ts = _parse_timestamp_from_filename(trace_file.name)
        if ts is None:
            continue
        if cutoff is not None and ts < cutoff:
            continue

        for span in _load_spans(trace_file):
            attrs = span.get("attributes", {})
            session_id = attrs.get(SESSION_ATTR)
            model = attrs.get("entity.2.name")

            if not session_id or not model:
                continue

            for event in span.get("events", []):
                if event.get("name") != "metadata":
                    continue
                event_attrs = event.get("attributes", {})
                for key in _TOKEN_KEYS:
                    val = event_attrs.get(key, 0)
                    if isinstance(val, (int, float)):
                        aggregated[session_id][model][key] += int(val)

    rows = []
    for session_id in sorted(aggregated.keys()):
        for model in sorted(aggregated[session_id].keys()):
            tokens = aggregated[session_id][model]
            rows.append(
                {
                    "session": session_id,
                    "model": model,
                    "prompt_tokens": tokens["prompt_tokens"],
                    "cache_read_input_tokens": tokens["cache_read_input_tokens"],
                    "cache_creation_input_tokens": tokens["cache_creation_input_tokens"],
                    "completion_tokens": tokens["completion_tokens"],
                    "total_tokens": tokens["total_tokens"],
                }
            )
    return rows


def format_table(rows):
    # type: (List[Dict]) -> str
    """Render *rows* as a plain-text ASCII table."""
    if not rows:
        return "No trace files found for the given time window."

    col_keys = [
        "session",
        "model",
        "prompt_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
        "completion_tokens",
        "total_tokens",
    ]

    widths = [len(h) for h in _HEADERS]
    for row in rows:
        for i, key in enumerate(col_keys):
            widths[i] = max(widths[i], len(str(row[key])))

    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    fmt = "| " + " | ".join("{{:<{}}}".format(w) for w in widths) + " |"

    lines = [sep, fmt.format(*_HEADERS), sep]
    for row in rows:
        lines.append(fmt.format(*[str(row[k]) for k in col_keys]))
    lines.append(sep)
    return "\n".join(lines)