"""Discover evals already recorded on a source fact in the Okahu eval store.

Read-only: lists candidate eval templates for the fact level, queries which ones
have a stored label for the fact's id(s), and returns eval-spec dicts shaped like
the test generator's injected evals. Non-fatal end to end.
"""
import logging
import os
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import requests

from monocle_test_tools.evals.okahu_eval import OkahuEval, DEFAULT_EVAL_TIME_PAD_SECONDS
from monocle_test_tools.evals.okahu_filtered_eval import OkahuFilteredEval, has_label

logger = logging.getLogger(__name__)

_SUPPORTED_EVAL_SOURCES = ("okahu",)


class _DiscoverySkipped(Exception):
    """Internal signal that discovery cannot proceed; carries a reason string."""


def _derive_query_inputs(spans, mapped_fact_name: str) -> Tuple[str, List[str], str, str]:
    """Return (workflow_name, fact_ids, start_time_iso, end_time_iso) from spans.

    Reuses OkahuEval.get_fact_map + enumerate_fact_ids for the id extraction.
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

    starts = [s.start_time for s in spans if s.start_time]
    ends = [s.end_time for s in spans if s.end_time]
    if not starts or not ends:
        raise _DiscoverySkipped("spans lack start/end times")
    pad_ns = int(os.getenv("OKAHU_EVAL_TIME_PAD_SECONDS", DEFAULT_EVAL_TIME_PAD_SECONDS)) * 1e9
    start_ns = min(starts) - pad_ns
    end_ns = max(ends) + pad_ns
    start_iso = datetime.fromtimestamp(start_ns / 1e9, timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    end_iso = datetime.fromtimestamp(end_ns / 1e9, timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    return workflow_name, fact_ids, start_iso, end_iso


def _list_candidate_evals(mapped_fact_name: str, api_key: str, eval_base: str) -> List[str]:
    """List built-in eval template names configured for the mapped fact level.

    The eval-query API rejects (HTTP 400) any eval_name not registered for the
    fact, so discovery must query only names returned here.
    """
    url = f"{eval_base.rstrip('/')}/v1/eval/templates"
    resp = requests.get(url, headers={"x-api-key": api_key},
                        params={"fact_name": mapped_fact_name}, timeout=60)
    resp.raise_for_status()
    templates = resp.json().get("templates", []) or []
    names = []
    for t in templates:
        name = t.get("name")
        # Keep templates whose group_by matches the fact level (mirrors verify_eval_template_exists).
        if name and (t.get("group_by") in (None, mapped_fact_name)):
            names.append(name)
    return names


def _query_fact_evals(client, workflow_name: str, fact_ids: List[str],
                      eval_names: List[str], *, fact_name: str,
                      start_time: str, end_time: str) -> List[dict]:
    """One batched POST to /evals/query for all candidate eval names.

    The eval-query API accepts the full name list in a single request and returns
    one row per (fact_id, eval_name), so discovery issues one call rather than one
    per template.
    """
    url = f"{client.api_base}/v1/workflows/{workflow_name}/evals/query"
    body = {
        "eval_names": eval_names,
        "fact_name": fact_name,
        "fact_ids": [str(f).removeprefix("0x") for f in fact_ids],
        "start_time": start_time,
        "end_time": end_time,
    }
    return list(client._paginate_post(url, body))


def discover_fact_evals(spans, *, fact_name: str = "traces",
                        eval_source: str = "okahu") -> Tuple[List[dict], Optional[str]]:
    """Query the Okahu eval store for evals already recorded on the fact.

    Returns (specs, note): note is None when specs were found, otherwise a
    human-readable reason for the generated-code comment + stderr warning.
    Never raises.
    """
    if eval_source not in _SUPPORTED_EVAL_SOURCES:
        return [], f"eval discovery skipped: unsupported eval_source '{eval_source}'"

    api_key = (os.getenv("OKAHU_API_KEY") or "").strip()
    eval_base = (os.getenv("OKAHU_EVALUATION_ENDPOINT") or "").strip()
    if not api_key:
        return [], "eval discovery skipped: OKAHU_API_KEY not configured"
    if not eval_base:
        return [], "eval discovery skipped: OKAHU_EVALUATION_ENDPOINT not configured"

    try:
        mapped = OkahuEval._map_fact_name(fact_name)
        workflow_name, fact_ids, start_time, end_time = _derive_query_inputs(spans, mapped)
        candidates = _list_candidate_evals(mapped, api_key, eval_base)
        if not candidates:
            return [], "No existing evals found on this fact"
        client = OkahuFilteredEval.from_env()
        rows = _query_fact_evals(
            client, workflow_name, fact_ids, candidates,
            fact_name=mapped, start_time=start_time, end_time=end_time)

        specs: List[dict] = []
        seen = set()
        for row in rows:
            if not has_label(row):
                continue
            name = row.get("eval_name")
            if not name or name in seen:
                continue
            seen.add(name)
            specs.append({
                "criteria": name,
                "expected": (row.get("eval_result") or {}).get("label"),
                "fact_name": fact_name,             # user-friendly; runtime maps it
                "eval_type": "builtin",
                "_discovered": True,
                "_discovered_fact_id": str(row.get("fact_id", "")).removeprefix("0x"),
            })
    except _DiscoverySkipped as exc:
        return [], f"eval discovery skipped: {exc}"
    except (requests.RequestException, RuntimeError, ValueError) as exc:
        logger.warning("Eval discovery failed: %s", exc)
        return [], f"eval discovery skipped: {exc}"

    if not specs:
        return [], "No existing evals found on this fact"
    return specs, None
