"""Dress a red-team scenario in a tested attack framing, using DeepTeam's templates.

Each vendored template (``third_party/deepteam/prompts``, Apache 2.0) gives
``enhance(input)``, a prompt that rewrites a baseline attack into its framing, and
``is_<x>(response)``, a prompt that checks the rewrite still shows that framing.
``dress`` applies both with DeepTeam's own discipline: up to three attempts, then the
plain baseline rather than a rewrite that failed its own check.
"""
import json
import re
from importlib import import_module
from typing import Any, Optional

ATTEMPTS = 3

# delivery -> (template module, class, check method, key in the check's JSON answer)
TEMPLATES = {
    "authority": ("authority_escalation", "AuthorityEscalationTemplate", "is_valid_authority", "is_valid_authority"),
    "urgency": ("emotional_manipulation", "EmotionalManipulationTemplate", "is_valid_emotional_attack", "is_valid_context"),
    "permission": ("permission_escalation", "PermissionEscalationTemplate", "is_valid_permission", "is_valid_permission"),
    "roleplay": ("roleplay", "RoleplayTemplate", "is_roleplay", "is_roleplay"),
    "probing": ("prompt_probing", "PromptProbingTemplate", "is_prompt_probing", "is_prompt_probing"),
}


def dress(model: Any, delivery: str, baseline: str, attempts: int = ATTEMPTS) -> str:
    """The baseline attack rewritten into ``delivery``'s framing, or the baseline itself."""
    if delivery not in TEMPLATES or model is None:
        return baseline
    module, class_name, check_name, key = TEMPLATES[delivery]
    template = getattr(import_module(f"{__package__}.third_party.deepteam.prompts.{module}"), class_name)
    for _ in range(attempts):
        rewritten = _answer(model, template.enhance(baseline)).get("input")
        if rewritten and _answer(model, getattr(template, check_name)(rewritten)).get(key) is True:
            return rewritten
    return baseline


def _answer(model: Any, prompt: str) -> dict:
    """The JSON object in the model's reply, or {} when there is none."""
    reply = model.invoke(prompt)
    text = getattr(reply, "content", reply)
    match = re.search(r"\{.*\}", str(text), re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def deliveries() -> list[str]:
    return list(TEMPLATES)


def template_for(delivery: str) -> Optional[str]:
    return TEMPLATES[delivery][0] if delivery in TEMPLATES else None
