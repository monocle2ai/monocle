"""Token usage summary from local Monocle trace files.

Reads .monocle/monocle_trace_*.json files and aggregates token counts
per date and model, displayed as a table in the terminal.
"""

import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from typing import List, Dict

MONOCLE_DIR = Path.home() / ".monocle"

# Token attribute keys extracted from the "metadata" event
_TOKEN_KEYS = [
    "prompt_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
    "completion_tokens",
    "total_tokens",
]

# Column header labels (same order as _TOKEN_KEYS)
_HEADERS = ["Date", "Model", "Input", "Cache Read", "Cache Create", "Output", "Total"]


def _parse_timestamp_from_filename(name):
    # type: (str) -> Optional[datetime]
    """Extract a UTC datetime from a filename ending like _2025-11-30_19.47.49.json"""
    # strip extension
    if name.endswith(".json"):
        stem = name[:-5]
    else:
        stem = name
    # last two underscore segments are the date and time parts
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
    """Return the earliest datetime to include, or None for 'all'."""
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
    return None  # "all" or unrecognised → no cutoff


def _load_spans(path):
    # type: (Path) -> List[Dict]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def summarize(time_window="all", monocle_dir=None):
    # type: (str, Optional[Path]) -> List[Dict]
    """Aggregate token usage from trace files in *monocle_dir*.

    Returns a list of row dicts sorted by date then model, each with keys:
    date, model, prompt_tokens, cache_read_input_tokens,
    cache_creation_input_tokens, completion_tokens, total_tokens.
    """
    if monocle_dir is None:
        monocle_dir = MONOCLE_DIR

    cutoff = _window_cutoff(time_window)

    # aggregated[date_str][model][token_key] = cumulative int
    aggregated = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    if not monocle_dir.exists():
        return []

    for trace_file in sorted(monocle_dir.glob("monocle_trace_*.json")):
        ts = _parse_timestamp_from_filename(trace_file.name)
        if ts is None:
            continue
        if cutoff is not None and ts < cutoff:
            continue

        date_str = ts.strftime("%Y-%m-%d")

        for span in _load_spans(trace_file):
            model = span.get("attributes", {}).get("entity.2.name")
            if not model:
                continue

            for event in span.get("events", []):
                if event.get("name") != "metadata":
                    continue
                attrs = event.get("attributes", {})
                for key in _TOKEN_KEYS:
                    val = attrs.get(key, 0)
                    if isinstance(val, (int, float)):
                        aggregated[date_str][model][key] += int(val)

    rows = []
    for date_str in sorted(aggregated.keys()):
        for model in sorted(aggregated[date_str].keys()):
            tokens = aggregated[date_str][model]
            rows.append(
                {
                    "date": date_str,
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
        "date",
        "model",
        "prompt_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
        "completion_tokens",
        "total_tokens",
    ]

    # compute column widths from headers and data
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