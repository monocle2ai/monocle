"""Turns Monocle spans into judge evidence."""

import ast
import datetime
import json
import re

from opentelemetry.sdk.trace import ReadableSpan
from pydantic import BaseModel

from monocle_test_tools.evals.monocle.template import EvidenceScope


class Evidence(BaseModel):
    trace_id: str
    text: str

ENTITY_KEY_PREFIX = "entity."
ENTITY_COUNT_KEY = "entity.count"
EPOCH = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)


def to_trace_file_shape(spans: list[ReadableSpan]) -> dict:
    """Return {"trace_id", "spans"} with each span in the dict shape of an exported trace file."""
    span_dicts = [_span_to_dict(span) for span in spans]
    return {"trace_id": _trace_id_of(spans[0]), "spans": span_dicts}


def _span_to_dict(span: ReadableSpan) -> dict:
    span_dict = {
        "span_name": span.name,
        "start_time": _iso_time_of(span.start_time),
        "end_time": _iso_time_of(span.end_time),
        "status": _status_of(span),
        "span_id": format(span.get_span_context().span_id, "016x"),
        "trace_id": _trace_id_of(span),
        "attributes": _fold_entities(dict(span.attributes or {})),
        "events": [_event_to_dict(event) for event in span.events],
    }
    if span.parent is not None:
        span_dict["parent_id"] = format(span.parent.span_id, "016x")
    return span_dict


def _fold_entities(attributes: dict) -> dict:
    """Group flat entity.N.key attributes into an ordered entity list; entity.count is kept."""
    folded = {}
    entities_by_position: dict[int, dict] = {}
    for key, value in attributes.items():
        if _is_entity_field(key):
            position, field = _entity_position_and_field(key)
            entities_by_position.setdefault(position, {})[field] = _listed(value)
        else:
            folded[key] = _listed(value)
    if entities_by_position:
        folded["entity"] = [entities_by_position[position] for position in sorted(entities_by_position)]
    return folded


def _is_entity_field(key: str) -> bool:
    if not key.startswith(ENTITY_KEY_PREFIX):
        return False
    if key == ENTITY_COUNT_KEY:
        return False
    key_parts = key.split(".", 2)
    if len(key_parts) < 3:
        return False
    return key_parts[1].isdigit()


def _entity_position_and_field(key: str) -> tuple[int, str]:
    _, position_text, field = key.split(".", 2)
    return int(position_text), field


def _event_to_dict(event) -> dict:
    return {
        "name": event.name,
        "timestamp": _iso_time_of(event.timestamp),
        "attributes": {key: _listed(value) for key, value in (event.attributes or {}).items()},
    }


def _listed(value):
    if isinstance(value, tuple):
        return list(value)
    return value


def _status_of(span: ReadableSpan) -> dict:
    return {"code": span.status.status_code.name.lower(), "message": span.status.description}


def _iso_time_of(nanoseconds: int | None) -> str | None:
    if nanoseconds is None:
        return None
    microseconds = (nanoseconds + 500) // 1000
    moment = EPOCH + datetime.timedelta(microseconds=microseconds)
    return moment.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _trace_id_of(span: ReadableSpan) -> str:
    return format(span.get_span_context().trace_id, "032x")


FINISH_REASON_MAP: dict[str, str] = {
    "STOP": "stop",
    "stop": "stop",
    "end_turn": "stop",
    "FUNCTION_CALL": "tool_call",
    "tool_calls": "tool_call",
    "tool_use": "tool_call",
    "MAX_TOKENS": "length",
    "length": "length",
    "max_tokens": "length",
    "safety": "error",
}
UNKNOWN_FINISH_REASON = "unknown"
INFERENCE_SPAN_TYPES = frozenset(("inference", "inference.framework"))
EMPTY_SENTINEL = "__empty__"
EMPTY_SENTINEL_FIELD_BY_EVENT = {"data.input": "input", "data.output": "response"}
EMPTY_VALUES = ("", [], {})


def normalize_spans(spans: list[dict]) -> None:
    """Rewrite each span in place into the canonical form the judge expects."""
    for span in spans:
        _normalize_span(span)


def _normalize_span(span: dict) -> None:
    finish_reason = canonicalize_finish_reason(_finish_reason_of(span))
    derived_subtype = derive_subtype(span, finish_reason)
    if derived_subtype is not None:
        span["attributes"]["span.subtype"] = derived_subtype
    unify_inference_type(span)
    normalize_stylistic_subtype(span)
    standardize_output_field(span)
    unwrap_output_value(span)
    unify_mcp_entity_prefix(span)
    apply_empty_sentinel(span)
    normalize_root_parent_id(span)


def canonicalize_finish_reason(raw_value: str | None) -> str | None:
    """Return the canonical finish reason, or 'unknown' for an unrecognised one; None stays None."""
    if raw_value is None:
        return None
    return FINISH_REASON_MAP.get(raw_value, UNKNOWN_FINISH_REASON)


def derive_subtype(span: dict, canonical_finish_reason: str | None) -> str | None:
    """Return the subtype an inference span lacks, or None when it has one or none can be derived."""
    attributes = span.get("attributes", {})
    if attributes.get("span.type", "") not in INFERENCE_SPAN_TYPES:
        return None
    if "span.subtype" in attributes:
        return None
    if canonical_finish_reason == "tool_call" and _has_tool_entity(attributes):
        return _subtype_of_tool_call_inference(span)
    if canonical_finish_reason == "stop":
        return "turn_end"
    return None


def _has_tool_entity(attributes: dict) -> bool:
    for entity in attributes.get("entity", []):
        if entity.get("type", "").startswith("tool."):
            return True
    return False


def _subtype_of_tool_call_inference(span: dict) -> str:
    if _has_transfer_to_in_output(span):
        return "delegation"
    return "tool_call"


def _has_transfer_to_in_output(span: dict) -> bool:
    event = _get_event(span, "data.output")
    if event is None:
        return False
    response = event.get("attributes", {}).get("response")
    if isinstance(response, str):
        return "transfer_to_" in response
    if isinstance(response, list):
        return any(isinstance(entry, str) and "transfer_to_" in entry for entry in response)
    return False


def unify_inference_type(span: dict) -> None:
    """Rewrite span type inference.framework to inference."""
    attributes = span.get("attributes", {})
    if attributes.get("span.type") == "inference.framework":
        attributes["span.type"] = "inference"


def normalize_stylistic_subtype(span: dict) -> None:
    """Give every tool invocation span that has a subtype the subtype content_generation."""
    attributes = span.get("attributes", {})
    if attributes.get("span.type") != "agentic.tool.invocation":
        return
    if "span.subtype" in attributes:
        attributes["span.subtype"] = "content_generation"


def standardize_output_field(span: dict) -> None:
    """Rename the data.output attribute 'output' to 'response' unless 'response' exists."""
    event = _get_event(span, "data.output")
    if event is None:
        return
    attributes = event.get("attributes", {})
    if "output" in attributes and "response" not in attributes:
        attributes["response"] = attributes.pop("output")


def unwrap_output_value(span: dict) -> None:
    """Replace a one-element response list with its only element."""
    event = _get_event(span, "data.output")
    if event is None:
        return
    attributes = event.get("attributes", {})
    response = attributes.get("response")
    if isinstance(response, list) and len(response) == 1:
        attributes["response"] = response[0]


def unify_mcp_entity_prefix(span: dict) -> None:
    """Rewrite entity types starting with fastmcp. to start with mcp."""
    for entity in span.get("attributes", {}).get("entity", []):
        entity_type = entity.get("type", "")
        if entity_type.startswith("fastmcp."):
            entity["type"] = entity_type.replace("fastmcp.", "mcp.", 1)


def apply_empty_sentinel(span: dict) -> None:
    """Replace an empty input or response value with the '__empty__' sentinel."""
    for event in span.get("events", []):
        _replace_empty_value_with_sentinel(event)


def _replace_empty_value_with_sentinel(event: dict) -> None:
    field = EMPTY_SENTINEL_FIELD_BY_EVENT.get(event.get("name", ""))
    if field is None:
        return
    attributes = event.get("attributes", {})
    if field not in attributes:
        return
    if _is_empty_value(attributes[field]):
        attributes[field] = EMPTY_SENTINEL


def _is_empty_value(value) -> bool:
    for empty_value in EMPTY_VALUES:
        if value == empty_value:
            return True
    return False


def normalize_root_parent_id(span: dict) -> None:
    """Make the parent_id of a root span an explicit None."""
    if _is_root_parent_id(span.get("parent_id")):
        span["parent_id"] = None


def _is_root_parent_id(parent_id) -> bool:
    return parent_id is None or parent_id == 0 or parent_id == "0"


def _finish_reason_of(span: dict) -> str | None:
    metadata_event = _get_event(span, "metadata")
    if metadata_event is None:
        return None
    return metadata_event.get("attributes", {}).get("finish_reason")


def _get_event(span: dict, event_name: str) -> dict | None:
    for event in span.get("events", []):
        if event.get("name") == event_name:
            return event
    return None


ENTITY_TYPE_PREFIXES = ("agent.", "tool.")
ENTITY_TYPES_EXACT = frozenset(("mcp.server",))
DROPPED_SPAN_TYPES = frozenset(("workflow", "inference.modelapi"))


def prepare_trace(trace: dict, allowed_events: set[str] | None = None) -> tuple[dict, dict]:
    """Return ({"trace_id", "spans"}, side_lookup) keeping only the fields the judge needs.

    Workflow and model-api spans are dropped and their children re-rooted.
    allowed_events limits which events survive; None keeps every judge-relevant event.
    The side lookup, keyed by span id, holds the decision span id, start time and finish
    reason and type of every span that has any of them.
    """
    spans = trace.get("spans", [])
    side_lookup = _side_lookup_of(spans)
    dropped_span_ids = _dropped_span_ids_of(spans)
    prepared_spans = []
    for span in spans:
        if span["span_id"] not in dropped_span_ids:
            prepared_spans.append(_prepared_span(span, dropped_span_ids, allowed_events))
    return {"trace_id": trace["trace_id"], "spans": prepared_spans}, side_lookup


def _dropped_span_ids_of(spans: list[dict]) -> set[str]:
    dropped_span_ids = set()
    for span in spans:
        if span.get("attributes", {}).get("span.type") in DROPPED_SPAN_TYPES:
            dropped_span_ids.add(span["span_id"])
    return dropped_span_ids


def _side_lookup_of(spans: list[dict]) -> dict:
    side_lookup = {}
    for span in spans:
        span_id = span.get("span_id")
        if not span_id:
            continue
        entry = _side_lookup_entry_of(span)
        if entry:
            side_lookup[span_id] = entry
    return side_lookup


def _side_lookup_entry_of(span: dict) -> dict:
    entry = {}
    decision_span_id = span.get("attributes", {}).get("inference.decision.span.id")
    if decision_span_id is not None:
        entry["inference_decision_span_id"] = decision_span_id
    if span.get("start_time") is not None:
        entry["start_time"] = span["start_time"]
    for event in span.get("events", []):
        if event.get("name") == "metadata":
            entry.update(_finish_fields_of(event.get("attributes", {})))
    return entry


def _finish_fields_of(metadata_attributes: dict) -> dict:
    finish_fields = {}
    if "finish_reason" in metadata_attributes:
        finish_fields["finish_reason"] = metadata_attributes["finish_reason"]
    if "finish_type" in metadata_attributes:
        finish_fields["finish_type"] = metadata_attributes["finish_type"]
    return finish_fields


def _prepared_span(span: dict, dropped_span_ids: set[str], allowed_events: set[str] | None) -> dict:
    parent_id = span.get("parent_id")
    if parent_id in dropped_span_ids:
        parent_id = None
    return {
        "span_id": span["span_id"],
        "parent_id": parent_id,
        "attributes": _kept_attributes(span.get("attributes", {})),
        "events": _kept_events(span.get("events", []), allowed_events),
    }


def _kept_attributes(attributes: dict) -> dict:
    kept = {}
    if "span.type" in attributes:
        kept["span.type"] = attributes["span.type"]
    if "span.subtype" in attributes:
        kept["span.subtype"] = attributes["span.subtype"]
    if attributes.get("entity") is not None:
        kept_entities = _kept_entities(attributes["entity"])
        if kept_entities:
            kept["entity"] = kept_entities
    return kept


def _kept_entities(entities: list) -> list:
    kept_entities = []
    for entity in entities:
        if _is_kept_entity_type(entity.get("type", "")):
            kept_entities.append(_trimmed_entity(entity))
    return kept_entities


def _is_kept_entity_type(entity_type: str) -> bool:
    if entity_type.startswith(ENTITY_TYPE_PREFIXES):
        return True
    return entity_type in ENTITY_TYPES_EXACT


def _trimmed_entity(entity: dict) -> dict:
    entity_type = entity.get("type", "")
    trimmed = {"type": entity_type}
    if "name" in entity:
        trimmed["name"] = entity["name"]
    if "server_name" in entity:
        trimmed["server_name"] = entity["server_name"]
    if entity_type.startswith("tool.") and "description" in entity:
        trimmed["description"] = entity["description"]
    return trimmed


def _kept_events(events: list, allowed_events: set[str] | None) -> list:
    kept_events = []
    for event in events:
        if allowed_events is not None and event.get("name", "") not in allowed_events:
            continue
        kept_event = _trimmed_event(event)
        if kept_event is not None:
            kept_events.append(kept_event)
    return kept_events


def _trimmed_event(event: dict) -> dict | None:
    name = event.get("name", "")
    attributes = event.get("attributes", {})
    if name == "data.input" and "input" in attributes:
        return {"name": name, "attributes": {"input": attributes["input"]}}
    if name == "data.output" and "response" in attributes:
        return {"name": name, "attributes": {"response": attributes["response"]}}
    if name == "data.output" and "output" in attributes:
        return {"name": name, "attributes": {"output": attributes["output"]}}
    return None


TOOL_INVOCATION_SPAN_TYPE = "agentic.tool.invocation"
AGENT_INVOCATION_SPAN_TYPE = "agentic.invocation"
TOOL_CALL_SUBTYPE = "tool_call"
DELEGATION_SUBTYPE = "delegation"
TURN_END_SUBTYPE = "turn_end"


def apply_causal_parent_ids(spans: list[dict], side_lookup: dict) -> list[dict]:
    """Rewrite parent_id of tool and agent invocation spans from structural to causal linkage.

    A tool invocation points at the inference that decided to call it; an agent invocation
    points at the delegation or turn-end inference that led to it. Every other span keeps
    its structural parent. Mutates the spans in place and returns them.
    """
    span_by_id = {span["span_id"]: span for span in spans}
    preceding, children_of = _build_preceding_siblings(spans)
    for span in spans:
        if _is_root_parent_id(span.get("parent_id")):
            continue
        span_type = _span_type_of(span)
        if span_type == TOOL_INVOCATION_SPAN_TYPE:
            _resolve_tool_invocation(span, span_by_id, preceding, side_lookup)
        elif span_type == AGENT_INVOCATION_SPAN_TYPE:
            _resolve_agent_invocation(span, span_by_id, preceding, children_of)
    return spans


def _resolve_tool_invocation(span: dict, span_by_id: dict, preceding: dict, side_lookup: dict) -> None:
    """Re-parent the tool span onto the inference that decided to call it, when one is found."""
    tool_name = _entity_name_with_prefix(span, "tool.")
    if not tool_name:
        return
    decision_span_id = _verified_decision_span_id(span, tool_name, span_by_id, side_lookup)
    if decision_span_id is not None:
        span["parent_id"] = decision_span_id
        return
    match = _walk_for_inference(
        span, span_by_id, preceding, subtype=TOOL_CALL_SUBTYPE, contains=tool_name, walk_ancestors=True
    )
    if match:
        span["parent_id"] = match


def _verified_decision_span_id(span: dict, tool_name: str, span_by_id: dict, side_lookup: dict) -> str | None:
    """Return the span's inference.decision.span.id, retargeted off streaming children,
    when that inference's response names the tool; otherwise None."""
    decision_span_id = side_lookup.get(span["span_id"], {}).get("inference_decision_span_id")
    if not decision_span_id:
        return None
    decision_span_id = _retarget_streaming_child(decision_span_id, span_by_id)
    decision_span = span_by_id.get(decision_span_id)
    if decision_span and _response_contains(decision_span, tool_name):
        return decision_span_id
    return None


def _resolve_agent_invocation(span: dict, span_by_id: dict, preceding: dict, children_of: dict) -> None:
    """Point the agent span at the delegation inference that transferred to it, else at the
    nearest preceding turn_end inference; with neither, keep the structural parent."""
    match = _delegating_inference_id(span, span_by_id, preceding, children_of)
    if not match:
        match = _walk_for_inference(
            span, span_by_id, preceding, subtype=TURN_END_SUBTYPE, children_of=children_of
        )
    if match:
        span["parent_id"] = match


def _delegating_inference_id(span: dict, span_by_id: dict, preceding: dict, children_of: dict) -> str | None:
    agent_name = _entity_name_with_prefix(span, "agent.")
    if not agent_name:
        return None
    return _walk_for_inference(
        span,
        span_by_id,
        preceding,
        subtype=DELEGATION_SUBTYPE,
        contains=f"transfer_to_{agent_name}",
        walk_ancestors=True,
        children_of=children_of,
    )


def _build_preceding_siblings(spans: list[dict]) -> tuple[dict, dict]:
    """Return (preceding, children_of) built from the structural parent_ids.

    preceding maps a span id to the ids of its earlier siblings, nearest first.
    children_of maps a parent id to its child span ids in order.
    """
    children_of: dict = {}
    for span in spans:
        parent_id = span.get("parent_id")
        if not _is_root_parent_id(parent_id):
            children_of.setdefault(parent_id, []).append(span["span_id"])
    preceding = {}
    for child_ids in children_of.values():
        for position, child_id in enumerate(child_ids):
            preceding[child_id] = list(reversed(child_ids[:position]))
    return preceding, children_of


def _walk_for_inference(
    span: dict,
    span_by_id: dict,
    preceding: dict,
    subtype: str,
    contains: str | None = None,
    walk_ancestors: bool = False,
    children_of: dict | None = None,
) -> str | None:
    """Return the id of the nearest preceding inference with the subtype whose response holds `contains`, or None."""
    current = span
    while current is not None:
        match = _nearest_matching_sibling(current, span_by_id, preceding, subtype, contains, children_of)
        if match is not None:
            return match
        if not walk_ancestors:
            return None
        current = _structural_parent_of(current, span_by_id)
    return None


def _nearest_matching_sibling(
    span: dict, span_by_id: dict, preceding: dict, subtype: str, contains: str | None, children_of: dict | None
) -> str | None:
    for sibling_id in preceding.get(span["span_id"], []):
        sibling = span_by_id.get(sibling_id)
        if sibling is None:
            continue
        match = _matching_span_id_within(sibling_id, sibling, span_by_id, subtype, contains, children_of)
        if match is not None:
            return match
    return None


def _matching_span_id_within(
    sibling_id: str, sibling: dict, span_by_id: dict, subtype: str, contains: str | None, children_of: dict | None
) -> str | None:
    if _is_matching_inference(sibling, subtype, contains):
        return sibling_id
    if _is_inference_span(sibling):
        return None
    if children_of is not None and _span_type_of(sibling) == AGENT_INVOCATION_SPAN_TYPE:
        return _matching_child_id(children_of.get(sibling_id, []), span_by_id, subtype, contains)
    return None


def _matching_child_id(child_ids: list[str], span_by_id: dict, subtype: str, contains: str | None) -> str | None:
    for child_id in reversed(child_ids):
        child = span_by_id.get(child_id)
        if child is None:
            continue
        if _is_matching_inference(child, subtype, contains):
            return child_id
    return None


def _is_matching_inference(span: dict, subtype: str, contains: str | None) -> bool:
    if not _is_inference_span(span):
        return False
    if span.get("attributes", {}).get("span.subtype") != subtype:
        return False
    if contains and not _response_contains(span, contains):
        return False
    return True


def _structural_parent_of(span: dict, span_by_id: dict) -> dict | None:
    parent_id = span.get("parent_id")
    if parent_id is None:
        return None
    return span_by_id.get(parent_id)


def _entity_name_with_prefix(span: dict, type_prefix: str) -> str | None:
    for entity in span.get("attributes", {}).get("entity", []):
        if entity.get("type", "").startswith(type_prefix):
            return entity.get("name")
    return None


def _response_contains(span: dict, needle: str) -> bool:
    """Return whether the span's data.output response, as text, contains the needle."""
    event = _get_event(span, "data.output")
    if event is None:
        return False
    response = event.get("attributes", {}).get("response")
    if response is None:
        return False
    if not isinstance(response, str):
        response = str(response)
    return needle in response


def _retarget_streaming_child(decision_span_id: str, span_by_id: dict) -> str:
    """Return the parent's id when the decision span is a streaming child, an inference nested
    under an inference of the same subtype; otherwise the id unchanged."""
    target = span_by_id.get(decision_span_id)
    if target is None or not _is_inference_span(target):
        return decision_span_id
    parent = _structural_parent_of(target, span_by_id)
    if parent is None or not _is_inference_span(parent):
        return decision_span_id
    if _subtype_of(target) == _subtype_of(parent):
        return target["parent_id"]
    return decision_span_id


def _is_inference_span(span: dict) -> bool:
    return _span_type_of(span) in INFERENCE_SPAN_TYPES


def _span_type_of(span: dict) -> str:
    return span.get("attributes", {}).get("span.type", "")


def _subtype_of(span: dict) -> str | None:
    return span.get("attributes", {}).get("span.subtype")


ALWAYS_EXCLUDED_SPAN_TYPES = frozenset(("workflow", "inference.modelapi", "agentic.delegation", "generic", "custom"))


def filter_and_rewire_spans(spans: list[dict], span_types: list[str]) -> list[dict]:
    """Return the spans worth judging, each parent_id rewired to its nearest surviving ancestor.

    Drops infrastructure spans and streaming-child inferences, then every span whose type is
    not in span_types; asking for either inference spelling keeps both. A span whose whole
    ancestry is dropped becomes a root. Mutates the spans in place.
    """
    parent_map = _build_parent_map(spans)
    spans = _apply_filter_and_rewire(spans, _identify_always_excluded(spans), parent_map)
    requested_span_types = _with_both_inference_spellings(span_types)
    spans = _apply_filter_and_rewire(spans, _ids_outside_span_types(spans, requested_span_types), parent_map)
    return spans


def _with_both_inference_spellings(span_types: list[str]) -> set[str]:
    requested_span_types = set(span_types)
    if requested_span_types & INFERENCE_SPAN_TYPES:
        requested_span_types.update(INFERENCE_SPAN_TYPES)
    return requested_span_types


def _ids_outside_span_types(spans: list[dict], span_types: set[str]) -> set[str]:
    return {span["span_id"] for span in spans if _span_type_of(span) not in span_types}


def _build_parent_map(spans: list[dict]) -> dict[str, str | None]:
    """Return {span_id: parent_id} for every span, with None for roots."""
    parent_map = {}
    for span in spans:
        parent_id = span.get("parent_id")
        if _is_root_parent_id(parent_id):
            parent_map[span["span_id"]] = None
        else:
            parent_map[span["span_id"]] = parent_id
    return parent_map


def _identify_always_excluded(spans: list[dict]) -> set[str]:
    """Return the ids of infrastructure spans and streaming-child inferences."""
    span_by_id = {span["span_id"]: span for span in spans}
    excluded_ids = set()
    for span in spans:
        if _span_type_of(span) in ALWAYS_EXCLUDED_SPAN_TYPES:
            excluded_ids.add(span["span_id"])
        elif _is_streaming_child(span, span_by_id):
            excluded_ids.add(span["span_id"])
    return excluded_ids


def _is_streaming_child(span: dict, span_by_id: dict) -> bool:
    if not _is_inference_span(span):
        return False
    parent_id = span.get("parent_id")
    if _is_root_parent_id(parent_id):
        return False
    parent = span_by_id.get(parent_id)
    if parent is None:
        return False
    if not _is_inference_span(parent):
        return False
    return _subtype_of(span) == _subtype_of(parent)


def _apply_filter_and_rewire(spans: list[dict], excluded_ids: set[str], parent_map: dict) -> list[dict]:
    """Return the spans not excluded, each parent_id pointing at a surviving span or None."""
    if not excluded_ids:
        return spans
    surviving_ids = {span["span_id"] for span in spans} - excluded_ids
    surviving_spans = []
    for span in spans:
        if span["span_id"] in excluded_ids:
            continue
        _rewire_to_surviving_parent(span, parent_map, surviving_ids)
        surviving_spans.append(span)
    return surviving_spans


def _rewire_to_surviving_parent(span: dict, parent_map: dict, surviving_ids: set[str]) -> None:
    parent_id = span.get("parent_id")
    if _is_root_parent_id(parent_id):
        return
    if parent_id in surviving_ids:
        return
    span["parent_id"] = _find_surviving_ancestor(parent_id, parent_map, surviving_ids)


def _find_surviving_ancestor(start_id: str, parent_map: dict, surviving_ids: set[str]) -> str | None:
    """Return the nearest ancestor of start_id, itself included, that survives; None when the walk ends or loops."""
    current_id = start_id
    visited_ids = set()
    while current_id is not None:
        if current_id in surviving_ids:
            return current_id
        if current_id in visited_ids:
            return None
        visited_ids.add(current_id)
        current_id = parent_map.get(current_id)
    return None


CANONICAL_ROLE_ALIASES = {
    "user": {"user", "human"},
    "assistant": {"assistant", "bot", "ai", "gpt", "model"},
    "system": {"system", "meta", "context"},
    "tool": {"tool", "tools", "function", "tool_results", "tool_function", "call_id"},
}
DEFAULT_ACCEPTED_ROLES = frozenset(("user", "assistant", "system", "tool"))
PYTHON_REPR_ROLE_MARKER = "Role(value="
PYTHON_REPR_ROLE_PATTERN = re.compile(r"Role\(value=['\"](\w+)['\"]\)")
INFERENCE_SPAN_TYPE = "inference"


def _canonical_role_by_alias() -> dict[str, str]:
    canonical_role_by_alias = {}
    for canonical_role, aliases in CANONICAL_ROLE_ALIASES.items():
        for alias in aliases:
            canonical_role_by_alias[alias] = canonical_role
    return canonical_role_by_alias


_ALIAS_TO_CANONICAL = _canonical_role_by_alias()


def normalize_and_filter_roles(
    spans: list[dict], accepted_roles: set[str] | None = None, span_types: list[str] | None = None
) -> list[dict]:
    """Rewrite the input and response of inference spans to canonical roles, keeping only accepted_roles.

    accepted_roles defaults to user, assistant, system and tool. When span_types asks for
    inference spans only (or is omitted), every parent_id is set to None. Mutates the spans
    in place and returns them.
    """
    if accepted_roles is None:
        accepted_roles = set(DEFAULT_ACCEPTED_ROLES)
    for span in spans:
        if _span_type_of(span) == INFERENCE_SPAN_TYPE:
            _normalize_inference_events(span, accepted_roles)
    if set(span_types or []) <= INFERENCE_SPAN_TYPES:
        _clear_parent_ids(spans)
    return spans


def _normalize_inference_events(span: dict, accepted_roles: set[str]) -> None:
    for event in span.get("events", []):
        event_name = event.get("name")
        if event_name == "data.input":
            _normalize_input_attribute(event.get("attributes", {}), accepted_roles)
        elif event_name == "data.output":
            _normalize_response_attribute(event.get("attributes", {}), accepted_roles)


def _normalize_input_attribute(event_attributes: dict, accepted_roles: set[str]) -> None:
    if "input" not in event_attributes:
        return
    event_attributes["input"] = normalize_input_array(event_attributes["input"], accepted_roles)


def _normalize_response_attribute(event_attributes: dict, accepted_roles: set[str]) -> None:
    if "response" not in event_attributes:
        return
    response = normalize_output_response(event_attributes["response"], accepted_roles)
    if response is None:
        event_attributes.pop("response")
    else:
        event_attributes["response"] = response


def _clear_parent_ids(spans: list[dict]) -> None:
    for span in spans:
        span["parent_id"] = None


def resolve_role(role: str) -> str:
    """Return the canonical role for an alias; an unknown role comes back lower-cased."""
    return _ALIAS_TO_CANONICAL.get(role.lower(), role.lower())


def parse_python_repr_input(input_string: str) -> list[dict[str, str]]:
    """Return the single-role messages of a Python-repr message list, e.g. "[{Role(value='user'): 'hi'}]".

    Roles come back as plain strings and None content as ''. Unparseable input gives [].
    """
    role_quoted = PYTHON_REPR_ROLE_PATTERN.sub(r"'\1'", input_string)
    try:
        parsed = ast.literal_eval(role_quoted)
    except (ValueError, SyntaxError):
        return []
    if not isinstance(parsed, list):
        return []
    messages = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        if len(item) != 1:
            continue
        role = next(iter(item))
        content = item[role]
        messages.append({str(role): str(content) if content is not None else ""})
    return messages


def normalize_input_array(input_array, accepted_roles: set[str]) -> list:
    """Return the input messages with canonical roles, dropping those whose role is not accepted.

    Messages with no identifiable role are kept as they are. An input that is not a
    message list comes back wrapped in a one-element list; None gives [].
    """
    if input_array is None:
        return []
    elements = _input_elements_of(input_array)
    if elements is None:
        return [input_array]
    messages = []
    for element in elements:
        role_and_content = _role_and_content_of_message(element)
        if role_and_content is None:
            messages.append(element)
            continue
        role, content = role_and_content
        canonical_role = resolve_role(role)
        if canonical_role in accepted_roles:
            messages.append({canonical_role: content})
    return messages


def _input_elements_of(input_array) -> list | None:
    if isinstance(input_array, list):
        return input_array
    if not isinstance(input_array, str):
        return None
    if PYTHON_REPR_ROLE_MARKER in input_array:
        return parse_python_repr_input(input_array)
    try:
        parsed = json.loads(input_array)
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(parsed, list):
        return parsed
    return None


def _role_and_content_of_message(message) -> tuple[str, object] | None:
    if isinstance(message, dict):
        return _single_role_and_content(message)
    if isinstance(message, str):
        parsed_object = _json_object_of(message)
        if parsed_object is None:
            return None
        return _single_role_and_content(parsed_object)
    return None


def _single_role_and_content(single_entry: dict) -> tuple[str, object] | None:
    if len(single_entry) != 1:
        return None
    role = next(iter(single_entry))
    return str(role), single_entry[role]


def _json_object_of(text: str) -> dict | None:
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(parsed, dict):
        return parsed
    return None


def normalize_output_response(response, accepted_roles: set[str]):
    """Return the response with a canonical role key, None when its role is not accepted.

    A one-element list is unwrapped first. A response with no identifiable role comes back unchanged.
    """
    if isinstance(response, list) and len(response) == 1:
        response = response[0]
    role_and_content = _role_and_content_of_response(response)
    if role_and_content is None:
        return response
    role, content = role_and_content
    canonical_role = resolve_role(role)
    if canonical_role not in accepted_roles:
        return None
    return {canonical_role: content}


def _role_and_content_of_response(response) -> tuple[str, object] | None:
    if isinstance(response, dict):
        return _single_string_role_and_content(response)
    if isinstance(response, str):
        parsed_object = _json_object_of(response)
        if parsed_object is None:
            return None
        return _single_string_role_and_content(parsed_object)
    return None


def _single_string_role_and_content(single_entry: dict) -> tuple[str, object] | None:
    if len(single_entry) != 1:
        return None
    role = next(iter(single_entry))
    if not isinstance(role, str):
        return None
    return role, single_entry[role]


MULTI_EVIDENCE_PREAMBLE = """\
You are evaluating an AI agent's execution trace.

The trace contains spans linked by parent_id into a causal tree that shows the agent's execution flow. The root span contains the user's original request and the agent's final response. Child spans reveal how the agent arrived at that response — which agents were invoked, what tools were called, and what data was retrieved or generated.

Key fields:
- entity.name and entity.description identify which agent or tool produced each span
- span.type and span.subtype indicate the kind of step (agent invocation, tool call, LLM inference, delegation)
- System messages in inference spans contain agent instructions
- Tool invocation inputs and outputs show the data the agent worked with

Here is the trace data:"""

THIN_PREAMBLE = """\
You are evaluating an AI model's response.

The trace contains inference spans, each representing one LLM call. Each span's input contains the conversation messages with their roles and the output contains the model's response.

Here is the trace data:"""

EVENT_NAME_BY_SCOPE_EVENT = {"input": "data.input", "output": "data.output"}
LAST_START_TIME = "\xff"
def preamble_for(span_types: list[str]) -> str:
    """Return the thin preamble when span_types asks for inference spans only, else the multi-evidence one."""
    if set(span_types) <= INFERENCE_SPAN_TYPES:
        return THIN_PREAMBLE
    return MULTI_EVIDENCE_PREAMBLE


def build_evidence(spans: list[ReadableSpan], trace_id: str, scope: EvidenceScope) -> Evidence | None:
    """Return the judge evidence for one trace: preprocessed, filtered to the scope, behind its preamble.

    Returns None when no span is left after filtering.
    """
    trace = _evidence_trace(spans, scope)
    if not trace["spans"]:
        return None
    return Evidence(trace_id=trace_id, text=serialize_evidence([trace], preamble_for(scope.span_types)))


def serialize_evidence(traces: list[dict], instructions: str) -> str:
    """Return one compact JSON document holding the instructions and every trace."""
    payload = {"instructions": instructions.strip(), "traces": traces}
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _evidence_trace(spans: list[ReadableSpan], scope: EvidenceScope) -> dict:
    trace = to_trace_file_shape(spans)
    normalize_spans(trace["spans"])
    prepared, side_lookup = prepare_trace(trace, _allowed_events_of(scope))
    evidence_spans = apply_causal_parent_ids(prepared["spans"], side_lookup)
    evidence_spans = _filtered_by_span_types(evidence_spans, scope.span_types)
    evidence_spans = normalize_and_filter_roles(evidence_spans, _accepted_roles_of(scope), span_types=scope.span_types)
    evidence_spans = _sorted_by_start_time(evidence_spans, side_lookup)
    return {"trace_id": prepared["trace_id"], "spans": evidence_spans}


def _allowed_events_of(scope: EvidenceScope) -> set[str] | None:
    if not scope.events:
        return None
    return {EVENT_NAME_BY_SCOPE_EVENT[event] for event in scope.events}


def _filtered_by_span_types(spans: list[dict], span_types: list[str]) -> list[dict]:
    if not span_types:
        return spans
    return filter_and_rewire_spans(spans, span_types)


def _accepted_roles_of(scope: EvidenceScope) -> set[str]:
    if not scope.roles:
        return set(DEFAULT_ACCEPTED_ROLES)
    return set(scope.roles)


def _sorted_by_start_time(spans: list[dict], side_lookup: dict) -> list[dict]:
    timed_spans = sorted(_timed_span(side_lookup, position, span) for position, span in enumerate(spans))
    return [span for _, _, span in timed_spans]


def _timed_span(side_lookup: dict, position: int, span: dict) -> tuple[str, int, dict]:
    return _start_time_of(side_lookup, span), position, span


def _start_time_of(side_lookup: dict, span: dict) -> str:
    return side_lookup.get(span["span_id"], {}).get("start_time", LAST_START_TIME)

