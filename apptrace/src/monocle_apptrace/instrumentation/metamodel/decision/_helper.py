"""Shared helpers for decision / classification model instrumentation.

Decision models (e.g. TypeSafe jev) are non-generative: a call returns a label /
score / yes-no answer with probabilities, not generated text. These helpers are
SDK-agnostic; each SDK supplies a `DecisionAdapter` with the extraction functions.
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional

from monocle_apptrace.instrumentation.common.constants import (
    DECISION_MODEL_PREFIX,
    LLM_MODEL_PREFIX,
    MODEL_TYPE_COMPLETION,
    MODEL_TYPE_DECISION,
)
from monocle_apptrace.instrumentation.metamodel.finish_types import (
    map_decision_finish_reason_to_finish_type,
)


@dataclass(frozen=True)
class DecisionAdapter:
    """Per-SDK extraction functions consumed by `make_decision_inference`."""
    provider_name: str
    get_inference_endpoint: Callable[[Any], Optional[str]]
    get_model: Callable[[dict], Optional[str]]
    extract_input: Callable[[dict], Any]
    extract_answers: Callable[[dict], Any]
    get_usage: Callable[[dict], dict]
    model_type: Optional[str] = None


def model_type(adapter: DecisionAdapter) -> str:
    """The adapter's model type; completion when not set."""
    return adapter.model_type or MODEL_TYPE_COMPLETION


def model_entity_type(adapter: DecisionAdapter, model_name: Optional[str]) -> str:
    prefix = DECISION_MODEL_PREFIX if model_type(adapter) == MODEL_TYPE_DECISION else LLM_MODEL_PREFIX
    return prefix + (model_name or "unknown")


def decision_finish_reason(arguments: dict) -> str:
    """A decision response has no finish_reason: synthesize one from success/exception."""
    return "error" if arguments.get("exception") is not None else "success"


def decision_finish_type(arguments: dict) -> Optional[str]:
    return map_decision_finish_reason_to_finish_type(decision_finish_reason(arguments))


def usage_metadata(usage: dict) -> dict:
    """Normalize token counts to the prompt/completion/total shape monocle records
    for inference spans, so decision-model tokens roll up like LLM tokens."""
    usage = usage or {}
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    if input_tokens is None and output_tokens is None:
        total_tokens = None
    else:
        total_tokens = (input_tokens or 0) + (output_tokens or 0)
    return {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": total_tokens,
    }
