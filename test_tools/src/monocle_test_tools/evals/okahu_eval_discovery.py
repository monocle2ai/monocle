"""Discover evals already recorded on a source fact in the Okahu eval store.

Read-only: one blind ``/evals/query`` call by workflow enumerates every eval on
the fact (builtin + custom); each labeled result becomes an eval-spec dict shaped
like the test generator's injected evals. Non-fatal end to end.
"""
import logging
import os
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import requests

from monocle_test_tools.evals.okahu_eval import OkahuEval
from monocle_test_tools.evals.okahu_filtered_eval import OkahuFilteredEval, has_label

logger = logging.getLogger(__name__)

# A result row whose eval_id starts with this prefix comes from a user-supplied
# custom template. Okahu does not store custom templates, so the generated
# assertion must be commented out with a request for the template path rather
# than run by name. (The custom row's eval_name can collide with a builtin one,
# e.g. "hallucination", so the eval_id prefix — not the name — is the signal.)
_CUSTOM_EVAL_ID_PREFIX = "custom_evaluation__"

# /evals/query defaults to a 90-day time window when none is supplied, which
# silently excludes evals recorded on older traces. We derive the window from the
# loaded spans' own start/end times and pad both ends generously so no eval on the
# fact is missed. The fact_ids are the precise filter, so a wide window is safe.
_TIME_WINDOW_PAD_SECONDS = 24 * 60 * 60  # 1 day


class _DiscoverySkipped(Exception):
    """Internal signal that discovery cannot proceed; carries a reason string."""


def _iso_utc(epoch_seconds: float) -> str:
    """Millisecond ISO-8601 UTC stamp (e.g. '2026-08-09T21:20:32.393Z')."""
    return (datetime.fromtimestamp(epoch_seconds, tz=timezone.utc)
            .isoformat(timespec="milliseconds").replace("+00:00", "Z"))


def _span_time_window(spans) -> Tuple[Optional[str], Optional[str]]:
    """Padded (start_time, end_time) covering every span, or (None, None).

    OTel span start_time/end_time are nanoseconds since epoch. Returns (None, None)
    when no span carries a usable time, in which case the query omits the window.
    """
    times = []
    for span in spans:
        for attr in ("start_time", "end_time"):
            t = getattr(span, attr, None)
            if t:
                times.append(t)
    if not times:
        return None, None
    lo = min(times) / 1e9 - _TIME_WINDOW_PAD_SECONDS
    hi = max(times) / 1e9 + _TIME_WINDOW_PAD_SECONDS
    return _iso_utc(lo), _iso_utc(hi)


def _derive_query_inputs(spans, mapped_fact_name: str) -> Tuple[str, List[str], Optional[str], Optional[str]]:
    """Return (workflow_name, fact_ids, start_time, end_time) from spans.

    Reuses OkahuEval.get_fact_map + enumerate_fact_ids for the id extraction, and
    derives the query time window from the spans' own start/end times.
    Raises _DiscoverySkipped when the workflow name or fact ids cannot be found.
    """
    if not spans:
        raise _DiscoverySkipped("no spans available")

    workflow_name = None
    for span in spans:
        wf = span.attributes.get("workflow.name")
        if wf:
            workflow_name = wf
            break
    if not workflow_name:
        raise _DiscoverySkipped("workflow.name not found in spans")

    evaluator = OkahuEval(eval_options={})
    fact_ids = evaluator.enumerate_fact_ids(filtered_spans=spans, fact_name=mapped_fact_name)
    if not fact_ids:
        raise _DiscoverySkipped(f"no fact ids for fact_name='{mapped_fact_name}'")
    start_time, end_time = _span_time_window(spans)
    return workflow_name, fact_ids, start_time, end_time


def _query_fact_evals(client, workflow_name: str, fact_ids: List[str], *, fact_name: str,
                      start_time: Optional[str] = None, end_time: Optional[str] = None) -> List[dict]:
    """Blind ``/evals/query`` by workflow: return every eval row for the fact ids.

    The endpoint enumerates all builtin and custom evals recorded on the fact and
    returns each with its result (if any). A span-derived start_time/end_time is
    included so evals on older traces aren't dropped by the 90-day default window.
    """
    url = f"{client.api_base}/v1/workflows/{workflow_name}/evals/query"
    body = {
        "fact_name": fact_name,
        "fact_ids": [str(f).removeprefix("0x") for f in fact_ids],
    }
    if start_time and end_time:
        body["start_time"] = start_time
        body["end_time"] = end_time
    return list(client._paginate_post(url, body))


def discover_fact_evals(spans, *, fact_name: str = "traces") -> Tuple[List[dict], Optional[str]]:
    """Query the Okahu eval store for evals already recorded on the fact.

    Returns (specs, note): ``note`` is None when specs were found, otherwise a
    human-readable reason for the generated-code comment + stderr warning.
    Never raises.

    Provider selection is handled by the caller (``BaseEval`` registry dispatch);
    this module is the Okahu implementation reached via ``OkahuEval.discover_fact_evals``.

    A row whose eval_id starts with ``custom_evaluation__`` comes from an unsaved
    custom template; its spec is tagged ``_discovered_custom`` so the generator
    emits a commented-out assertion asking for the template path.
    """
    # Only the API key must be set. The Okahu endpoints default to prod (users
    # never set them); developers override OKAHU_API_ENDPOINT / _EVALUATION_ENDPOINT.
    api_key = (os.getenv("OKAHU_API_KEY") or "").strip()
    if not api_key:
        return [], "eval discovery skipped: OKAHU_API_KEY not configured"

    try:
        mapped = OkahuEval._map_fact_name(fact_name)
        workflow_name, fact_ids, start_time, end_time = _derive_query_inputs(spans, mapped)
        client = OkahuFilteredEval.from_env()
        rows = _query_fact_evals(client, workflow_name, fact_ids, fact_name=mapped,
                                 start_time=start_time, end_time=end_time)

        specs: List[dict] = []
        seen = set()
        for row in rows:
            if not has_label(row):
                continue
            name = row.get("eval_name")
            if not name:
                continue
            is_custom = str(row.get("eval_id") or "").startswith(_CUSTOM_EVAL_ID_PREFIX)
            # De-dup on (name, provenance): a builtin and a custom template can
            # share a name (e.g. "hallucination") and must both be kept.
            dedupe = (name, is_custom)
            if dedupe in seen:
                continue
            seen.add(dedupe)
            specs.append({
                "criteria": name,
                "expected": (row.get("eval_result") or {}).get("label"),
                "fact_name": fact_name,             # user-friendly; runtime maps it
                "eval_type": "custom" if is_custom else "builtin",
                "_discovered": True,
                "_discovered_fact_id": str(row.get("fact_id", "")).removeprefix("0x"),
                "_discovered_custom": is_custom,
            })
    except _DiscoverySkipped as exc:
        return [], f"eval discovery skipped: {exc}"
    except (requests.RequestException, RuntimeError, ValueError) as exc:
        logger.warning("Eval discovery failed: %s", exc)
        return [], f"eval discovery skipped: {exc}"

    if not specs:
        return [], "No existing evals found on this fact"
    return specs, None
