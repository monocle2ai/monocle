import json
import logging
import os
import re

from pydantic import BaseModel, ConfigDict

__all__ = ["TemplateError", "EvalTemplate", "EvidenceScope", "LlmConfig", "OutputField", "load", "coherence_warnings"]

logger = logging.getLogger(__name__)


class TemplateError(ValueError):
    """A template breaks one of the rules; the message says which."""


class EvidenceScope(BaseModel):
    span_types: list[str]
    roles: list[str]
    events: list[str]

    @classmethod
    def default(cls) -> "EvidenceScope":
        return cls(
            span_types=[
                "agentic.turn", "inference", "inference.framework", "agentic.invocation",
                "agentic.tool.invocation", "agentic.mcp.invocation", "embedding",
                "embedding.modelapi", "retrieval", "retrieval.embedding",
            ],
            roles=["system", "user", "assistant", "tool"],
            events=["input", "output"],
        )


class LlmConfig(BaseModel):
    model: str
    params: dict[str, int | float] = {}
    min_confidence: float | None = None
    context_tokens: int | None = None

    @property
    def provider(self) -> str:
        """The text before the colon in `model`: "openai" for "openai:gpt-4.1-mini"."""
        return self.model.partition(":")[0]


class OutputField(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str
    type: str | None = None
    enums: list[str] | None = None
    items: dict | None = None
    anyOf: list | None = None


class EvalTemplate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    eval_prompt: str
    structure_output: dict[str, OutputField]
    evidence_scope: EvidenceScope
    llm: LlmConfig

    @property
    def labels(self) -> list[str]:
        return self.structure_output["label"].enums

TEMPLATE_KEYS = {"name", "eval_prompt", "structure_output", "evidence_scope", "llm"}
NAME_PATTERN = re.compile(r"^[a-z0-9]+(_[a-z0-9]+)*$")
MAX_NAME_LENGTH = 50
LABEL_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
FIELD_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
FIELD_TYPES = {"string", "number", "integer", "boolean", "array", "object", "enum", "anyOf"}
FIELD_KEYS = {"description", "type", "enum", "enums", "items", "anyOf"}
SPAN_TYPES = set(EvidenceScope.default().span_types)
ROLES = set(EvidenceScope.default().roles)
EVENTS = set(EvidenceScope.default().events)
SCOPE_KEYS = {"span_types", "roles", "events"}
LLM_KEYS = {"model", "params", "min_confidence", "context_tokens"}
LLM_PARAMS = {"temperature", "top_p", "top_k", "max_tokens", "seed"}


def load(template: str | os.PathLike | dict) -> EvalTemplate:
    """Validate a template given as a file path or a dict and log each coherence warning; raise TemplateError on the first broken rule."""
    template_dict = _read_file(template) if isinstance(template, (str, os.PathLike)) else template
    _check_top_level_keys(template_dict)
    _check_name(template_dict.get("name"))
    _check_eval_prompt(template_dict.get("eval_prompt"))
    structure_output = _check_structure_output(template_dict.get("structure_output"))
    llm = _check_llm(template_dict.get("llm"))
    scope = _check_scope(template_dict.get("evidence_scope"))
    eval_template = EvalTemplate(
        name=template_dict["name"],
        eval_prompt=template_dict["eval_prompt"],
        structure_output={field_name: OutputField(**field) for field_name, field in structure_output.items()},
        evidence_scope=scope,
        llm=llm,
    )
    for warning in coherence_warnings(eval_template):
        logger.warning("Template '%s': %s", eval_template.name, warning)
    return eval_template


def coherence_warnings(template: EvalTemplate) -> list[str]:
    """Plain-sentence warnings where the eval_prompt asks for evidence the evidence_scope leaves out."""
    prompt = template.eval_prompt.lower()
    scope = template.evidence_scope
    checks = [_tool_warning, _mcp_warning, _retrieval_warning, _user_question_warning, _system_prompt_warning]
    warnings = [check(prompt, scope) for check in checks]
    return [warning for warning in warnings if warning is not None]


def _mentions(prompt: str, *phrases: str) -> bool:
    return any(phrase in prompt for phrase in phrases)


def _tool_warning(prompt: str, scope: EvidenceScope) -> str | None:
    asks_for_tools = _mentions(prompt, "tool call", "tool use", "tool invocation")
    scope_covers_tools = "agentic.tool.invocation" in scope.span_types or "tool" in scope.roles
    if asks_for_tools and not scope_covers_tools:
        return (
            "eval_prompt mentions tool calls but evidence_scope excludes "
            "'agentic.tool.invocation' spans and the 'tool' role."
        )
    return None


def _mcp_warning(prompt: str, scope: EvidenceScope) -> str | None:
    asks_for_mcp = _mentions(prompt, "mcp")
    scope_covers_mcp = "agentic.mcp.invocation" in scope.span_types
    if asks_for_mcp and not scope_covers_mcp:
        return "eval_prompt mentions MCP but evidence_scope excludes 'agentic.mcp.invocation' spans."
    return None


def _retrieval_warning(prompt: str, scope: EvidenceScope) -> str | None:
    asks_for_retrieval = _mentions(prompt, "retriev", "document", "context chunk", "rag")
    scope_covers_retrieval = any(span_type.startswith("retrieval") for span_type in scope.span_types)
    if asks_for_retrieval and not scope_covers_retrieval:
        return "eval_prompt mentions retrieval or documents but evidence_scope excludes 'retrieval' spans."
    return None


def _user_question_warning(prompt: str, scope: EvidenceScope) -> str | None:
    asks_for_user_question = _mentions(prompt, "user question", "user input", "the question", "user's question")
    scope_covers_user_question = "input" in scope.events and "user" in scope.roles
    if asks_for_user_question and not scope_covers_user_question:
        return (
            "eval_prompt refers to the user's question but evidence_scope excludes "
            "the 'input' event or the 'user' role."
        )
    return None


def _system_prompt_warning(prompt: str, scope: EvidenceScope) -> str | None:
    asks_for_system_prompt = _mentions(prompt, "system prompt", "system instruction")
    scope_covers_system_prompt = "system" in scope.roles
    if asks_for_system_prompt and not scope_covers_system_prompt:
        return "eval_prompt mentions the system prompt but evidence_scope excludes the 'system' role."
    return None


def _read_file(path: str | os.PathLike) -> dict:
    with open(path) as template_file:
        contents = json.load(template_file)
    if isinstance(contents, dict) and set(contents) == {"template"}:
        return contents["template"]
    return contents


def _check_top_level_keys(template: dict) -> None:
    if not isinstance(template, dict):
        raise TemplateError("The template must be a JSON object.")
    unknown_keys = set(template) - TEMPLATE_KEYS
    if unknown_keys:
        raise TemplateError(
            f"Unrecognized template field(s): {', '.join(sorted(unknown_keys))}. "
            f"Allowed fields: {', '.join(sorted(TEMPLATE_KEYS))}."
        )


def _check_name(name) -> None:
    if not isinstance(name, str) or not name:
        raise TemplateError("The template needs a 'name' field (a string).")
    if len(name) > MAX_NAME_LENGTH or not NAME_PATTERN.match(name):
        raise TemplateError(
            "The 'name' may only contain lower-case letters, numbers and single underscores, "
            "must not start or end with an underscore, and is at most 50 characters."
        )


def _check_eval_prompt(eval_prompt) -> None:
    if not isinstance(eval_prompt, str) or not eval_prompt.strip():
        raise TemplateError("The template needs a non-empty 'eval_prompt' field (a string).")


def _check_structure_output(structure_output) -> dict[str, dict]:
    """Return the fields with every allowed-values key spelled 'enums'."""
    if not isinstance(structure_output, dict) or not structure_output:
        raise TemplateError("The template needs a 'structure_output' field with at least one field.")
    if "label" not in structure_output:
        raise TemplateError("'structure_output' needs a 'label' field.")
    if "explanation" not in structure_output:
        raise TemplateError("'structure_output' needs an 'explanation' field.")
    fields = {}
    for field_name, field in structure_output.items():
        _check_output_field(field_name, field)
        fields[field_name] = _spelled_with_enums(field)
    _check_label(fields["label"])
    return fields


def _spelled_with_enums(field: dict) -> dict:
    if "enum" not in field:
        return field
    spelled = {key: value for key, value in field.items() if key != "enum"}
    spelled["enums"] = field["enum"]
    return spelled


def _check_label(label: dict) -> None:
    enums = label.get("enums")
    if enums is None:
        raise TemplateError("The 'label' field needs 'enums' listing the labels the judge may give.")
    _check_enums("label", enums)
    if len(enums) != len(set(enums)):
        raise TemplateError("The 'label' field's 'enums' must not contain duplicate values.")
    for enum in enums:
        if not LABEL_PATTERN.match(enum):
            raise TemplateError(
                f"The label '{enum}' in the 'label' field's 'enums' is not valid. A label starts with a letter "
                "and contains only letters, numbers and underscores, up to 64 characters."
            )


def _check_output_field(field_name: str, field) -> None:
    if not FIELD_NAME_PATTERN.match(field_name):
        raise TemplateError(
            f"The field name '{field_name}' in 'structure_output' may only contain letters, numbers and underscores."
        )
    if not isinstance(field, dict):
        raise TemplateError(f"The field '{field_name}' in 'structure_output' must be an object.")
    unknown_keys = set(field) - FIELD_KEYS
    if unknown_keys:
        raise TemplateError(
            f"The field '{field_name}' has unrecognized key(s): {', '.join(sorted(unknown_keys))}. "
            f"Allowed keys: {', '.join(sorted(FIELD_KEYS))}."
        )
    description = field.get("description")
    if not isinstance(description, str) or not description.strip():
        raise TemplateError(f"The field '{field_name}' needs a non-empty 'description'.")
    if "type" in field and not _is_known_type(field["type"]):
        raise TemplateError(
            f"The field '{field_name}' has an unknown type '{field['type']}'. "
            f"Valid types: {', '.join(sorted(FIELD_TYPES))}."
        )
    if "enum" in field and "enums" in field:
        raise TemplateError(f"The field '{field_name}' has both 'enum' and 'enums'; use one.")
    for enums_key in ("enum", "enums"):
        if enums_key in field:
            _check_enums(field_name, field[enums_key])
    if "items" in field and not isinstance(field["items"], dict):
        raise TemplateError(f"The 'items' of field '{field_name}' must be an object.")
    if "anyOf" in field and not isinstance(field["anyOf"], list):
        raise TemplateError(f"The 'anyOf' of field '{field_name}' must be a list.")


def _is_known_type(field_type) -> bool:
    return isinstance(field_type, str) and field_type in FIELD_TYPES


def _check_enums(field_name: str, enums) -> None:
    if not isinstance(enums, list) or not enums or not all(isinstance(enum, str) and enum for enum in enums):
        raise TemplateError(f"The 'enums' of field '{field_name}' must be a list of non-empty strings.")


def _check_llm(llm) -> LlmConfig:
    if not isinstance(llm, dict) or not isinstance(llm.get("model"), str) or not llm["model"]:
        raise TemplateError("The template needs an 'llm.model' field (a string such as 'openai:gpt-4.1').")
    unknown_keys = set(llm) - LLM_KEYS
    if unknown_keys:
        raise TemplateError(
            f"Unrecognized llm field(s): {', '.join(sorted(unknown_keys))}. Allowed: {', '.join(sorted(LLM_KEYS))}."
        )
    _check_llm_params(llm.get("params", {}))
    _check_min_confidence(llm.get("min_confidence"))
    _check_context_tokens(llm["model"], llm.get("context_tokens"))
    return LlmConfig(**llm)


def _check_llm_params(params) -> None:
    if not isinstance(params, dict):
        raise TemplateError("'llm.params' must be an object, such as {\"temperature\": 0}.")
    unknown_params = set(params) - LLM_PARAMS
    if unknown_params:
        raise TemplateError(
            f"Unrecognized llm.params field(s): {', '.join(sorted(unknown_params))}. "
            f"Allowed: {', '.join(sorted(LLM_PARAMS))}."
        )
    for param_name, value in params.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TemplateError(f"'llm.params.{param_name}' must be a number.")


def _check_context_tokens(model: str, context_tokens) -> None:
    if context_tokens is None:
        return
    if isinstance(context_tokens, bool) or not isinstance(context_tokens, int) or context_tokens < 1:
        raise TemplateError("'llm.context_tokens' must be a whole number of tokens above 0.")
    if not (model == "local" or model.startswith("local:")):
        raise TemplateError("'llm.context_tokens' applies to a local model only ('local' or 'local:<name>').")


def _check_min_confidence(min_confidence) -> None:
    if min_confidence is None:
        return
    if isinstance(min_confidence, bool) or not isinstance(min_confidence, (int, float)):
        raise TemplateError("'llm.min_confidence' must be a number.")
    if not 0 < min_confidence <= 1:
        raise TemplateError("'llm.min_confidence' must be above 0 and at most 1.")


def _check_scope(scope) -> EvidenceScope:
    if scope is None:
        return EvidenceScope.default()
    if not isinstance(scope, dict):
        raise TemplateError("'evidence_scope' must be an object.")
    unknown_keys = set(scope) - SCOPE_KEYS
    if unknown_keys:
        raise TemplateError(
            f"Unrecognized evidence_scope field(s): {', '.join(sorted(unknown_keys))}. "
            f"Allowed: {', '.join(sorted(SCOPE_KEYS))}."
        )
    missing_keys = SCOPE_KEYS - set(scope)
    if missing_keys:
        raise TemplateError(f"'evidence_scope' is missing required field(s): {', '.join(sorted(missing_keys))}.")
    _check_scope_values("span_types", scope["span_types"], SPAN_TYPES)
    _check_scope_values("roles", scope["roles"], ROLES)
    _check_scope_values("events", scope["events"], EVENTS)
    return EvidenceScope(**scope)


def _check_scope_values(key: str, values, allowed: set[str]) -> None:
    if not isinstance(values, list) or not values or not all(isinstance(value, str) for value in values):
        raise TemplateError(f"'evidence_scope.{key}' must be a non-empty list of strings.")
    unknown_values = set(values) - allowed
    if unknown_values:
        raise TemplateError(
            f"Unknown value(s) in 'evidence_scope.{key}': {', '.join(sorted(unknown_values))}. "
            f"Allowed: {', '.join(sorted(allowed))}."
        )
