"""Optionally add OpenTelemetry GenAI attributes alongside the Monocle metamodel."""

import os
from typing import Any, Optional, Tuple

from monocle_apptrace.instrumentation.common.constants import SPAN_TYPES


_OPERATION_BY_SPAN_TYPE = {
    SPAN_TYPES.AGENTIC_TOOL_INVOCATION: "execute_tool",
    SPAN_TYPES.AGENTIC_MCP_INVOCATION: "execute_tool",
    SPAN_TYPES.AGENTIC_INVOCATION: "invoke_agent",
    SPAN_TYPES.AGENTIC_REQUEST: "invoke_agent",
    SPAN_TYPES.RETRIEVAL: "retrieval",
}

OTEL_GENAI_SEMCONV_ENV = "MONOCLE_OTEL_GENAI_SEMCONV"
OTEL_GENAI_SEMCONV_EXPORTER = "otlp-genai-semconv"
_otel_genai_semconv_enabled = False


def configure_otel_genai_semconv(
    setting: Any = None, exporter_names: Tuple[str, ...] = ()
) -> bool:
    """Resolve auto/true/false configuration and return the enabled state."""
    global _otel_genai_semconv_enabled

    value = setting
    if value is None:
        value = os.environ.get(OTEL_GENAI_SEMCONV_ENV, "auto")

    if isinstance(value, bool):
        enabled = value
    else:
        normalized = str(value).strip().lower()
        if normalized == "auto":
            enabled = OTEL_GENAI_SEMCONV_EXPORTER in exporter_names
        elif normalized in {"true", "1", "yes", "on"}:
            enabled = True
        elif normalized in {"false", "0", "no", "off"}:
            enabled = False
        else:
            raise ValueError(
                f"{OTEL_GENAI_SEMCONV_ENV} must be one of auto, true, or false"
            )

    _otel_genai_semconv_enabled = enabled
    return enabled


def _entity_indices(attributes: Any) -> Tuple[int, ...]:
    """Return every entity index present without imposing an arbitrary limit."""
    indices = set()
    for key in attributes:
        parts = str(key).split(".", 2)
        if len(parts) == 3 and parts[0] == "entity" and parts[1].isdigit():
            indices.add(int(parts[1]))

    entity_count = attributes.get("entity.count")
    if isinstance(entity_count, int) and entity_count > 0:
        indices.update(range(1, entity_count + 1))
    return tuple(sorted(indices))


def _find_entity(attributes: Any, type_prefixes: Tuple[str, ...]) -> Optional[str]:
    for index in _entity_indices(attributes):
        entity_type = attributes.get(f"entity.{index}.type")
        if not isinstance(entity_type, str) or not entity_type.startswith(type_prefixes):
            continue
        entity_name = attributes.get(f"entity.{index}.name")
        if entity_name is not None:
            return str(entity_name)
    return None


def _operation_for_span(attributes: Any) -> Optional[str]:
    span_type = attributes.get("span.type")
    operation = _OPERATION_BY_SPAN_TYPE.get(span_type)
    if operation is not None:
        return operation

    if isinstance(span_type, str) and (
        span_type == "embedding" or span_type.startswith("embedding.")
    ):
        return "embeddings"

    if span_type in {SPAN_TYPES.INFERENCE, SPAN_TYPES.INFERENCE_FRAMEWORK}:
        if _find_entity(attributes, ("model.embedding",)) is not None:
            return "embeddings"
        return "chat"
    return None


def _metadata_value(span: Any, *names: str) -> Optional[Any]:
    for event in getattr(span, "events", ()) or ():
        if getattr(event, "name", None) != "metadata":
            continue
        event_attributes = getattr(event, "attributes", None) or {}
        for name in names:
            value = event_attributes.get(name)
            if value is not None:
                return value
    return None


def _set_if_missing(span: Any, key: str, value: Optional[Any]) -> None:
    if value is None or key in (span.attributes or {}):
        return
    span.set_attribute(key, value)


def enrich_genai_attributes(span: Any) -> None:
    """Add standard GenAI attributes without replacing Monocle attributes."""
    if not _otel_genai_semconv_enabled:
        return

    attributes = span.attributes or {}
    operation = _operation_for_span(attributes)
    if operation is None:
        return

    _set_if_missing(span, "gen_ai.operation.name", operation)
    _set_if_missing(span, "gen_ai.workflow.name", attributes.get("workflow.name"))

    provider = next(
        (
            attributes.get(f"entity.{index}.provider_name")
            for index in _entity_indices(attributes)
            if attributes.get(f"entity.{index}.provider_name") is not None
        ),
        None,
    )
    _set_if_missing(span, "gen_ai.provider.name", provider)
    model_prefixes = ("model.embedding",) if operation == "embeddings" else ("model.llm",)
    _set_if_missing(span, "gen_ai.request.model", _find_entity(attributes, model_prefixes))

    input_tokens = _metadata_value(span, "prompt_tokens", "input_tokens")
    output_tokens = _metadata_value(span, "completion_tokens", "output_tokens")
    _set_if_missing(span, "gen_ai.usage.input_tokens", input_tokens)
    _set_if_missing(span, "gen_ai.usage.output_tokens", output_tokens)

    if operation == "execute_tool":
        _set_if_missing(span, "gen_ai.tool.name", _find_entity(attributes, ("tool.", "tool")))
    elif operation == "invoke_agent":
        _set_if_missing(span, "gen_ai.agent.name", _find_entity(attributes, ("agent.", "agentic.")))
