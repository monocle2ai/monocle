"""Dress a red-team scenario in a tested attack framing, using DeepTeam's templates.

Each vendored template (``third_party/deepteam/prompts``, Apache 2.0) gives
``enhance(input)``, a prompt that rewrites a baseline attack into its framing, and
``is_<x>(response)``, a prompt that checks the rewrite still shows that framing.
``dress`` applies both with DeepTeam's own discipline: up to three attempts, then the
plain baseline rather than a rewrite that failed its own check.
"""
import json
import logging
from typing import Any, Optional

from monocle_test_tools.scenarios.third_party.deepteam.prompts.authority_escalation import AuthorityEscalationTemplate
from monocle_test_tools.scenarios.third_party.deepteam.prompts.emotional_manipulation import EmotionalManipulationTemplate
from monocle_test_tools.scenarios.third_party.deepteam.prompts.permission_escalation import PermissionEscalationTemplate
from monocle_test_tools.scenarios.third_party.deepteam.prompts.prompt_probing import PromptProbingTemplate
from monocle_test_tools.scenarios.third_party.deepteam.prompts.roleplay import RoleplayTemplate

logger = logging.getLogger(__name__)

ATTEMPTS = 3

# delivery -> (template, its check method, the key in the check's JSON answer)
TEMPLATES = {
    "authority": (AuthorityEscalationTemplate, "is_valid_authority", "is_valid_authority"),
    "urgency": (EmotionalManipulationTemplate, "is_valid_emotional_attack", "is_valid_context"),
    "permission": (PermissionEscalationTemplate, "is_valid_permission", "is_valid_permission"),
    "roleplay": (RoleplayTemplate, "is_roleplay", "is_roleplay"),
    "probing": (PromptProbingTemplate, "is_prompt_probing", "is_prompt_probing"),
}


def dress(model: Any, delivery: str, baseline: str, attempts: int = ATTEMPTS) -> Optional[str]:
    """The baseline attack rewritten into ``delivery``'s framing, or None when no rewrite
    passed its check (or the delivery has no template, or there is no model)."""
    if delivery not in TEMPLATES or model is None:
        return None
    template, check_name, key = TEMPLATES[delivery]
    for attempt in range(1, attempts + 1):
        rewritten = _answer(model, template.enhance(baseline)).get("input")
        if rewritten and _answer(model, getattr(template, check_name)(rewritten)).get(key) is True:
            return rewritten
        logger.info("%s: attempt %d/%d did not pass DeepTeam's check", delivery, attempt, attempts)
    return None


def _answer(model: Any, prompt: str) -> dict:
    """The first JSON object in the model's reply, or {} when there is none."""
    reply = model.invoke(prompt)
    text = _text(getattr(reply, "content", reply))
    start = text.find("{")
    if start < 0:
        return {}
    try:
        value, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _text(content: Any) -> str:
    """Reply content as a string. Some providers return a list of content blocks."""
    if isinstance(content, list):
        return "\n".join(block.get("text", "") if isinstance(block, dict) else str(block) for block in content)
    return str(content)
