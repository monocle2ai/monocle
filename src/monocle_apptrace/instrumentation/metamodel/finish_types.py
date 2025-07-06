"""
This module provides common finish reason mappings and finish type enums
for different AI providers (OpenAI, Anthropic, Gemini).
"""

from enum import Enum


class FinishType(Enum):
    """Enum for standardized finish types across all AI providers."""
    SUCCESS = "success"
    TRUNCATED = "truncated"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"
    REFUSAL = "refusal"


# OpenAI finish reason mapping
OPENAI_FINISH_REASON_MAPPING = {
    "stop": FinishType.SUCCESS.value,
    "tool_calls": FinishType.SUCCESS.value,
    "function_call": FinishType.SUCCESS.value,  # deprecated but still possible
    "length": FinishType.TRUNCATED.value,
    "content_filter": FinishType.CONTENT_FILTER.value
}

# Anthropic finish reason mapping
ANTHROPIC_FINISH_REASON_MAPPING = {
    "end_turn": FinishType.SUCCESS.value,         # Natural completion
    "max_tokens": FinishType.TRUNCATED.value,     # Hit max_tokens limit
    "stop_sequence": FinishType.SUCCESS.value,    # Hit user stop sequence
    "tool_use": FinishType.SUCCESS.value,         # Tool use triggered
    "pause_turn": FinishType.SUCCESS.value,       # Paused for tool or server action
    "refusal": FinishType.REFUSAL.value,          # Refused for safety/ethics
}

# Gemini finish reason mapping
GEMINI_FINISH_REASON_MAPPING = {
    "STOP": FinishType.SUCCESS.value,
    "MAX_TOKENS": FinishType.TRUNCATED.value,
    "SAFETY": FinishType.CONTENT_FILTER.value,
    "RECITATION": FinishType.CONTENT_FILTER.value,
    "OTHER": FinishType.ERROR.value,
    "FINISH_REASON_UNSPECIFIED": None
}


def map_openai_finish_reason_to_finish_type(finish_reason):
    """Map OpenAI finish_reason to standardized finish_type."""
    if not finish_reason:
        return None
    return OPENAI_FINISH_REASON_MAPPING.get(finish_reason, None)


def map_anthropic_finish_reason_to_finish_type(finish_reason):
    """Map Anthropic stop_reason to standardized finish_type."""
    if not finish_reason:
        return None
    return ANTHROPIC_FINISH_REASON_MAPPING.get(finish_reason, None)


def map_gemini_finish_reason_to_finish_type(finish_reason):
    """Map Gemini finish_reason to standardized finish_type."""
    if not finish_reason:
        return None
    return GEMINI_FINISH_REASON_MAPPING.get(finish_reason, None)
