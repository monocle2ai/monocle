"""Adapter for TypeSafe's `system_one` decision model (jev).

Reads a `typesafe_sdk` `SystemOneResponse` for the shared decision-model processor.
`state`/`questions` may arrive positionally or as kwargs (the real caller uses kwargs).
"""

import logging
from typing import Any, Dict, Optional

from monocle_apptrace.instrumentation.common.constants import MODEL_TYPE_DECISION
from monocle_apptrace.instrumentation.common.utils import get_json_dumps
from monocle_apptrace.instrumentation.metamodel.decision._helper import DecisionAdapter

logger = logging.getLogger(__name__)

PROVIDER = "typesafe"


def _arg(arguments: dict, index: int, *kw_names: str):
    """Read a positional arg by index, or any of kw_names from kwargs (kwargs win)."""
    kwargs = arguments.get("kwargs") or {}
    for name in kw_names:
        if kwargs.get(name) is not None:
            return kwargs[name]
    args = arguments.get("args") or ()
    if len(args) > index:
        return args[index]
    return None


def get_inference_endpoint(instance) -> Optional[str]:
    try:
        for attr in ("base_url", "_base_url"):
            val = getattr(instance, attr, None)
            if val:
                return str(val)
        config = getattr(instance, "_config", None)
        if config is not None:
            val = getattr(config, "base_url", None)
            if val:
                return str(val)
    except Exception as e:
        logger.debug("typesafe get_inference_endpoint error: %s", e)
    return None


def get_model(arguments: dict) -> Optional[str]:
    """Resolved model from the response; fall back to the requested alias."""
    result = arguments.get("result")
    resolved = getattr(result, "model", None)
    if resolved:
        return str(resolved)
    requested = (arguments.get("kwargs") or {}).get("model")
    return str(requested) if requested else None


def _question_to_dict(question) -> dict:
    out = {}
    for attr in ("type", "instructions", "criteria"):
        if isinstance(question, dict):
            val = question.get(attr)
        else:
            val = getattr(question, attr, None)
        if val is not None:
            out[attr] = val
    return out


def extract_input(arguments: dict):
    """Serialize the evaluated `state` plus each named question (type/instructions/criteria)."""
    try:
        state = _arg(arguments, 0, "state")
        questions = _arg(arguments, 1, "questions") or {}
        messages = []
        if state is not None:
            messages.append({"state": state})
        if isinstance(questions, dict):
            for name, question in questions.items():
                messages.append({name: _question_to_dict(question)})
        return [get_json_dumps(message) for message in messages]
    except Exception as e:
        logger.warning("typesafe extract_input error: %s", e)
        return []


def _answer_to_dict(answer) -> dict:
    atype = getattr(answer, "type", None)
    if atype == "choice":
        return {"type": "choice", "choice": getattr(answer, "choice", None),
                "confidence": getattr(answer, "confidence", None),
                "probabilities": dict(getattr(answer, "probabilities", {}) or {})}
    if atype == "noul":
        return {"type": "noul", "noul": getattr(answer, "noul", None)}
    if atype == "score":
        return {"type": "score", "score": getattr(answer, "score", None),
                "confidence": getattr(answer, "confidence", None),
                "probabilities": dict(getattr(answer, "probabilities", {}) or {}),
                "legend": dict(getattr(answer, "legend", {}) or {})}
    if atype == "box":
        return {"type": "box", "box": str(getattr(answer, "box", None)),
                "confidence": getattr(answer, "confidence", None)}
    return {"type": atype}


def extract_answers(arguments: dict):
    """Serialize the SystemOneResponse answers dict (mixed answer types) to JSON."""
    try:
        result = arguments.get("result")
        answers = getattr(result, "answers", None)
        if not answers:
            return ""
        serialized = {name: _answer_to_dict(answer) for name, answer in answers.items()}
        return get_json_dumps(serialized)
    except Exception as e:
        logger.warning("typesafe extract_answers error: %s", e)
        return None


def get_usage(arguments: dict) -> Dict[str, Any]:
    result = arguments.get("result")
    usage = getattr(result, "usage", None)
    if usage is None:
        return {}
    return {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
    }


TYPESAFE_ADAPTER = DecisionAdapter(
    provider_name=PROVIDER,
    get_inference_endpoint=get_inference_endpoint,
    get_model=get_model,
    extract_input=extract_input,
    extract_answers=extract_answers,
    get_usage=get_usage,
    model_type=MODEL_TYPE_DECISION,
)
