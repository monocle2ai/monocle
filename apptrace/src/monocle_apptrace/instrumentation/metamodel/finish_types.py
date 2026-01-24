"""
This module provides common finish reason mappings and finish type enums
for different AI providers (OpenAI, Anthropic, Gemini, LangChain, LlamaIndex, Azure AI Inference).
"""

from enum import Enum

class FinishType(Enum):
    """Enum for standardized finish types across all AI providers."""
    SUCCESS = "success"
    TRUNCATED = "truncated"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"
    TOOL_CALL_ERROR = "tool_call_error"
    REFUSAL = "refusal"
    RATE_LIMITED = "rate_limited"
    TOOL_CALL = "tool_call"

# OpenAI finish reason mapping
OPENAI_FINISH_REASON_MAPPING = {
    "stop": FinishType.SUCCESS.value,
    "tool_calls": FinishType.TOOL_CALL.value,
    "function_call": FinishType.TOOL_CALL.value,  # deprecated but still possible
    "length": FinishType.TRUNCATED.value,
    "content_filter": FinishType.CONTENT_FILTER.value
}

# Anthropic finish reason mapping
ANTHROPIC_FINISH_REASON_MAPPING = {
    "end_turn": FinishType.SUCCESS.value,         # Natural completion
    "max_tokens": FinishType.TRUNCATED.value,     # Hit max_tokens limit
    "stop_sequence": FinishType.SUCCESS.value,    # Hit user stop sequence
    "tool_use": FinishType.TOOL_CALL.value,       # Tool use triggered
    "pause_turn": FinishType.SUCCESS.value,       # Paused for tool or server action
    "refusal": FinishType.REFUSAL.value,          # Refused for safety/ethics
}

# Gemini finish reason mapping
GEMINI_FINISH_REASON_MAPPING = {
    "STOP": FinishType.SUCCESS.value,
    "FUNCTION_CALL": FinishType.TOOL_CALL.value,
    "MAX_TOKENS": FinishType.TRUNCATED.value,
    "SAFETY": FinishType.CONTENT_FILTER.value,
    "RECITATION": FinishType.CONTENT_FILTER.value,
    "MALFORMED_FUNCTION_CALL": FinishType.TOOL_CALL_ERROR.value,
    "OTHER": FinishType.ERROR.value,
    "FINISH_REASON_UNSPECIFIED": None
}

# LlamaIndex finish reason mapping
# LlamaIndex often wraps underlying provider responses, similar to LangChain
LLAMAINDEX_FINISH_REASON_MAPPING = {
    # Standard completion reasons
    "stop": FinishType.SUCCESS.value,
    "complete": FinishType.SUCCESS.value,
    "finished": FinishType.SUCCESS.value,
    "success": FinishType.SUCCESS.value,
    
    # Token limits
    "length": FinishType.TRUNCATED.value,
    "max_tokens": FinishType.TRUNCATED.value,
    "token_limit": FinishType.TRUNCATED.value,
    "truncated": FinishType.TRUNCATED.value,
    
    # Tool/function calling
    "tool_calls": FinishType.TOOL_CALL.value,
    "function_call": FinishType.TOOL_CALL.value,
    "agent_finish": FinishType.SUCCESS.value,
    
    # Content filtering and safety
    "content_filter": FinishType.CONTENT_FILTER.value,
    "safety": FinishType.CONTENT_FILTER.value,
    "filtered": FinishType.CONTENT_FILTER.value,
    
    # Errors
    "error": FinishType.ERROR.value,
    "failed": FinishType.ERROR.value,
    "exception": FinishType.ERROR.value,
    
    # Provider-specific reasons that might pass through LlamaIndex
    # OpenAI reasons
    "end_turn": FinishType.SUCCESS.value,  # Anthropic
    "stop_sequence": FinishType.SUCCESS.value,  # Anthropic
    "STOP": FinishType.SUCCESS.value,  # Gemini
    "SAFETY": FinishType.CONTENT_FILTER.value,  # Gemini
    "RECITATION": FinishType.CONTENT_FILTER.value,  # Gemini
    "OTHER": FinishType.ERROR.value,  # Gemini
}

# Azure AI Inference finish reason mapping
AZURE_AI_INFERENCE_FINISH_REASON_MAPPING = {
    # Standard completion reasons
    "stop": FinishType.SUCCESS.value,
    "completed": FinishType.SUCCESS.value,
    "finished": FinishType.SUCCESS.value,
    
    # Token limits
    "length": FinishType.TRUNCATED.value,
    "max_tokens": FinishType.TRUNCATED.value,
    "token_limit": FinishType.TRUNCATED.value,
    "max_completion_tokens": FinishType.TRUNCATED.value,
    
    # Tool/function calling
    "tool_calls": FinishType.TOOL_CALL.value,
    "function_call": FinishType.TOOL_CALL.value,

    # Content filtering and safety
    "content_filter": FinishType.CONTENT_FILTER.value,
    "content_filtered": FinishType.CONTENT_FILTER.value,
    "safety": FinishType.CONTENT_FILTER.value,
    "responsible_ai_policy": FinishType.CONTENT_FILTER.value,
    
    # Errors
    "error": FinishType.ERROR.value,
    "failed": FinishType.ERROR.value,
    "exception": FinishType.ERROR.value,
    "timeout": FinishType.ERROR.value,
    
    # Azure-specific reasons
    "model_error": FinishType.ERROR.value,
    "service_unavailable": FinishType.ERROR.value,
    "rate_limit": FinishType.ERROR.value,
}

# AWS Bedrock finish reason mapping
# Based on AWS Bedrock Converse API and model-specific APIs
BEDROCK_FINISH_REASON_MAPPING = {
    # Standard completion reasons
    "end_turn": FinishType.SUCCESS.value,           # Natural completion
    "stop": FinishType.SUCCESS.value,               # Hit stop sequence
    "stop_sequence": FinishType.SUCCESS.value,      # Stop sequence triggered
    "completed": FinishType.SUCCESS.value,          # Completion finished successfully
    
    # Token limits
    "max_tokens": FinishType.TRUNCATED.value,       # Hit max_tokens limit
    "length": FinishType.TRUNCATED.value,           # Token length limit
    "max_length": FinishType.TRUNCATED.value,       # Maximum length reached
    "token_limit": FinishType.TRUNCATED.value,      # Token limit reached
    
    # Tool/function calling
    "tool_use": FinishType.TOOL_CALL.value,           # Tool use triggered
    "function_call": FinishType.TOOL_CALL.value,      # Function call triggered

    # Content filtering and safety
    "content_filter": FinishType.CONTENT_FILTER.value,    # Content filtered
    "content_filtered": FinishType.CONTENT_FILTER.value,  # Content was filtered
    "safety": FinishType.CONTENT_FILTER.value,            # Safety filter triggered
    "guardrails": FinishType.CONTENT_FILTER.value,        # Bedrock guardrails triggered
    "blocked": FinishType.CONTENT_FILTER.value,           # Request blocked
    
    # Errors
    "error": FinishType.ERROR.value,                # General error
    "failed": FinishType.ERROR.value,               # Request failed
    "exception": FinishType.ERROR.value,            # Exception occurred
    "timeout": FinishType.ERROR.value,              # Request timeout
    "model_error": FinishType.ERROR.value,          # Model-specific error
    "service_unavailable": FinishType.ERROR.value,  # Service unavailable
    "throttled": FinishType.ERROR.value,            # Request throttled
    "rate_limit": FinishType.ERROR.value,           # Rate limit exceeded
    "validation_error": FinishType.ERROR.value,     # Validation error
    
    # Model-specific reasons (various Bedrock models)
    # Claude models via Bedrock
    "end_turn": FinishType.SUCCESS.value,           # Already defined above
    "max_tokens": FinishType.TRUNCATED.value,       # Already defined above
    "stop_sequence": FinishType.SUCCESS.value,      # Already defined above
    "tool_use": FinishType.TOOL_CALL.value,         # Already defined above

    # AI21 models via Bedrock
    "endoftext": FinishType.SUCCESS.value,          # AI21 end of text
    "length": FinishType.TRUNCATED.value,           # AI21 length limit
    
    # Cohere models via Bedrock
    "COMPLETE": FinishType.SUCCESS.value,           # Cohere completion
    "MAX_TOKENS": FinishType.TRUNCATED.value,       # Cohere max tokens
    "ERROR": FinishType.ERROR.value,                # Cohere error
    
    # Meta Llama models via Bedrock
    "stop": FinishType.SUCCESS.value,               # Already defined above
    "length": FinishType.TRUNCATED.value,           # Already defined above
    
    # Amazon Titan models via Bedrock
    "FINISH": FinishType.SUCCESS.value,             # Titan finish
    "LENGTH": FinishType.TRUNCATED.value,           # Titan length limit
    "CONTENT_FILTERED": FinishType.CONTENT_FILTER.value,  # Titan content filter
}

# LangChain finish reason mapping
# LangChain often wraps underlying provider responses, so we include common finish reasons
# that might appear in LangChain response objects
LANGCHAIN_FINISH_REASON_MAPPING = {
    # Standard completion reasons
    "stop": FinishType.SUCCESS.value,
    "complete": FinishType.SUCCESS.value,
    "finished": FinishType.SUCCESS.value,
    
    # Token limits
    "length": FinishType.TRUNCATED.value,
    "max_tokens": FinishType.TRUNCATED.value,
    "token_limit": FinishType.TRUNCATED.value,
    
    # Tool/function calling
    "tool_calls": FinishType.TOOL_CALL.value,
    "function_call": FinishType.TOOL_CALL.value,
    "tool_use": FinishType.TOOL_CALL.value,  # Anthropic tool use finish reason
    
    # Content filtering and safety
    "content_filter": FinishType.CONTENT_FILTER.value,
    "safety": FinishType.CONTENT_FILTER.value,
    "filtered": FinishType.CONTENT_FILTER.value,
    
    # Errors
    "error": FinishType.ERROR.value,
    "failed": FinishType.ERROR.value,
    "exception": FinishType.ERROR.value,
    
    # Provider-specific reasons that might pass through LangChain
    # OpenAI reasons
    "stop": FinishType.SUCCESS.value,  # Already defined above
    
    # Anthropic reasons
    "end_turn": FinishType.SUCCESS.value,
    "stop_sequence": FinishType.SUCCESS.value,
    
    # Gemini reasons
    "STOP": FinishType.SUCCESS.value,
    "SAFETY": FinishType.CONTENT_FILTER.value,
    "RECITATION": FinishType.CONTENT_FILTER.value,
    "OTHER": FinishType.ERROR.value,
}

TEAMSAI_FINISH_REASON_MAPPING = {
    "success": FinishType.SUCCESS.value,
    "error": FinishType.ERROR.value,
    "too_long": FinishType.TRUNCATED.value,
    "rate_limited": FinishType.RATE_LIMITED.value,
    "invalid_response": FinishType.ERROR.value,
}
# Haystack finish reason mapping
HAYSTACK_FINISH_REASON_MAPPING = {
    # Standard completion reasons
    "stop": FinishType.SUCCESS.value,
    "complete": FinishType.SUCCESS.value,
    "finished": FinishType.SUCCESS.value,

    # Token limits
    "length": FinishType.TRUNCATED.value,
    "max_tokens": FinishType.TRUNCATED.value,
    "token_limit": FinishType.TRUNCATED.value,

    # Tool/function calling
    "tool_calls": FinishType.TOOL_CALL.value,
    "function_call": FinishType.TOOL_CALL.value,
    "tool_use": FinishType.TOOL_CALL.value,  # Anthropic tool use finish reason

    # Content filtering and safety
    "content_filter": FinishType.CONTENT_FILTER.value,
    "safety": FinishType.CONTENT_FILTER.value,
    "filtered": FinishType.CONTENT_FILTER.value,

    # Errors
    "error": FinishType.ERROR.value,
    "failed": FinishType.ERROR.value,
    "exception": FinishType.ERROR.value,

    # Provider-specific reasons that might pass through LangChain
    # OpenAI reasons
    "stop": FinishType.SUCCESS.value,  # Already defined above

    # Anthropic reasons
    "end_turn": FinishType.SUCCESS.value,
    "stop_sequence": FinishType.SUCCESS.value,

    # Gemini reasons
    "STOP": FinishType.SUCCESS.value,
    "SAFETY": FinishType.CONTENT_FILTER.value,
    "RECITATION": FinishType.CONTENT_FILTER.value,
    "OTHER": FinishType.ERROR.value,
}

MISTRAL_FINISH_REASON_MAPPING = {
    "stop": FinishType.SUCCESS.value,
    "tool_calls": FinishType.TOOL_CALL.value,  # New category for tool calls
    "length": FinishType.TRUNCATED.value,
    # Mistral's API documentation does not explicitly mention other finish reasons like "content_filter" or "refusal".
    # However, in case of an API-level error, the response itself would likely be an HTTP error rather than a
    # successful response with a specific finish reason.
}

HUGGING_FACE_FINISH_REASON_MAPPING = {
    "stop": FinishType.SUCCESS.value,
    "tool_calls": FinishType.TOOL_CALL.value,  # New category for tool calls
    "length": FinishType.TRUNCATED.value,
    # Hugging Face's API documentation does not explicitly mention other finish reasons like "content_filter" or "refusal".
    # However, in case of an API-level error, the response itself would likely be an HTTP error rather than a
    # successful response with a specific finish reason.
}

ADK_FINISH_REASON_MAPPING = GEMINI_FINISH_REASON_MAPPING

LITELLM_FINISH_REASON_MAPPING = {
    "stop": FinishType.SUCCESS.value,
    "tool_calls": FinishType.TOOL_CALL.value,
    "function_call": FinishType.TOOL_CALL.value,
    "length": FinishType.TRUNCATED.value,
    "content_filter": FinishType.CONTENT_FILTER.value
}

# Microsoft Agent Framework finish reason mapping
# MS Agent Framework uses Azure OpenAI Assistants API which may not populate finish_reason in streaming
MSAGENT_FINISH_REASON_MAPPING = {
    "stop": FinishType.SUCCESS.value,
    "completed": FinishType.SUCCESS.value,
    "tool_calls": FinishType.TOOL_CALL.value,
    "function_call": FinishType.TOOL_CALL.value,
    "length": FinishType.TRUNCATED.value,
    "max_tokens": FinishType.TRUNCATED.value,
    "content_filter": FinishType.CONTENT_FILTER.value,
    "content_filtered": FinishType.CONTENT_FILTER.value,
}

def map_openai_finish_reason_to_finish_type(finish_reason):
    """Map OpenAI finish_reason to standardized finish_type."""
    if not finish_reason:
        return None
    return OPENAI_FINISH_REASON_MAPPING.get(finish_reason, None)


def map_msagent_finish_reason_to_finish_type(finish_reason):
    """Map Microsoft Agent Framework finish_reason to standardized finish_type.
    
    Note: Azure OpenAI Assistants API often returns None for finish_reason in streaming responses.
    Callers should provide a default finish_reason like 'stop' when None is encountered.
    """
    if not finish_reason:
        return None
    return MSAGENT_FINISH_REASON_MAPPING.get(finish_reason, None)


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


def map_langchain_finish_reason_to_finish_type(finish_reason):
    """Map LangChain finish_reason to standardized finish_type."""
    if not finish_reason:
        return None
    
    # Convert to lowercase for case-insensitive matching
    finish_reason_lower = finish_reason.lower() if isinstance(finish_reason, str) else str(finish_reason).lower()
    
    # Try direct mapping first
    if finish_reason in LANGCHAIN_FINISH_REASON_MAPPING:
        return LANGCHAIN_FINISH_REASON_MAPPING[finish_reason]
    
    # Try lowercase mapping
    if finish_reason_lower in LANGCHAIN_FINISH_REASON_MAPPING:
        return LANGCHAIN_FINISH_REASON_MAPPING[finish_reason_lower]
    
    # If no direct mapping, try to infer from common patterns
    if any(keyword in finish_reason_lower for keyword in ['stop', 'complete', 'success', 'done']):
        return FinishType.SUCCESS.value
    elif any(keyword in finish_reason_lower for keyword in ['length', 'token', 'limit', 'truncat']):
        return FinishType.TRUNCATED.value
    elif any(keyword in finish_reason_lower for keyword in ['filter', 'safety', 'block']):
        return FinishType.CONTENT_FILTER.value
    elif any(keyword in finish_reason_lower for keyword in ['error', 'fail', 'exception']):
        return FinishType.ERROR.value
    
    return None


def map_llamaindex_finish_reason_to_finish_type(finish_reason):
    """Map LlamaIndex finish_reason to standardized finish_type."""
    if not finish_reason:
        return None
    
    # Convert to lowercase for case-insensitive matching
    finish_reason_lower = finish_reason.lower() if isinstance(finish_reason, str) else str(finish_reason).lower()
    
    # Try direct mapping first
    if finish_reason in LLAMAINDEX_FINISH_REASON_MAPPING:
        return LLAMAINDEX_FINISH_REASON_MAPPING[finish_reason]
    
    # Try lowercase mapping
    if finish_reason_lower in LLAMAINDEX_FINISH_REASON_MAPPING:
        return LLAMAINDEX_FINISH_REASON_MAPPING[finish_reason_lower]
    
    # If no direct mapping, try to infer from common patterns
    if any(keyword in finish_reason_lower for keyword in ['stop', 'complete', 'success', 'done', 'finish']):
        return FinishType.SUCCESS.value
    elif any(keyword in finish_reason_lower for keyword in ['length', 'token', 'limit', 'truncat']):
        return FinishType.TRUNCATED.value
    elif any(keyword in finish_reason_lower for keyword in ['filter', 'safety', 'block']):
        return FinishType.CONTENT_FILTER.value
    elif any(keyword in finish_reason_lower for keyword in ['error', 'fail', 'exception']):
        return FinishType.ERROR.value
    
    return None


def map_azure_ai_inference_finish_reason_to_finish_type(finish_reason):
    """Map Azure AI Inference finish_reason to standardized finish_type."""
    if not finish_reason:
        return None
    
    # Convert to lowercase for case-insensitive matching
    finish_reason_lower = finish_reason.lower() if isinstance(finish_reason, str) else str(finish_reason).lower()
    
    # Try direct mapping first
    if finish_reason in AZURE_AI_INFERENCE_FINISH_REASON_MAPPING:
        return AZURE_AI_INFERENCE_FINISH_REASON_MAPPING[finish_reason]
    
    # Try lowercase mapping
    if finish_reason_lower in AZURE_AI_INFERENCE_FINISH_REASON_MAPPING:
        return AZURE_AI_INFERENCE_FINISH_REASON_MAPPING[finish_reason_lower]
    
    # If no direct mapping, try to infer from common patterns
    if any(keyword in finish_reason_lower for keyword in ['stop', 'complete', 'success', 'done', 'finish']):
        return FinishType.SUCCESS.value
    elif any(keyword in finish_reason_lower for keyword in ['length', 'token', 'limit', 'truncat']):
        return FinishType.TRUNCATED.value
    elif any(keyword in finish_reason_lower for keyword in ['filter', 'safety', 'block', 'responsible_ai', 'content_filter']):
        return FinishType.CONTENT_FILTER.value
    elif any(keyword in finish_reason_lower for keyword in ['error', 'fail', 'exception', 'timeout', 'unavailable', 'rate_limit']):
        return FinishType.ERROR.value
    
    return None


def map_bedrock_finish_reason_to_finish_type(finish_reason):
    """Map AWS Bedrock finish_reason/stopReason to standardized finish_type."""
    if not finish_reason:
        return None
    
    # Convert to lowercase for case-insensitive matching
    finish_reason_lower = finish_reason.lower() if isinstance(finish_reason, str) else str(finish_reason).lower()
    
    # Try direct mapping first
    if finish_reason in BEDROCK_FINISH_REASON_MAPPING:
        return BEDROCK_FINISH_REASON_MAPPING[finish_reason]
    
    # Try lowercase mapping
    if finish_reason_lower in BEDROCK_FINISH_REASON_MAPPING:
        return BEDROCK_FINISH_REASON_MAPPING[finish_reason_lower]
    
    # If no direct mapping, try to infer from common patterns
    if any(keyword in finish_reason_lower for keyword in ['stop', 'complete', 'success', 'done', 'finish', 'end_turn', 'endoftext']):
        return FinishType.SUCCESS.value
    elif any(keyword in finish_reason_lower for keyword in ['length', 'token', 'limit', 'truncat', 'max_tokens']):
        return FinishType.TRUNCATED.value
    elif any(keyword in finish_reason_lower for keyword in ['filter', 'safety', 'block', 'guardrails', 'content_filter']):
        return FinishType.CONTENT_FILTER.value
    elif any(keyword in finish_reason_lower for keyword in ['error', 'fail', 'exception', 'timeout', 'unavailable', 'rate_limit', 'throttled', 'validation']):
        return FinishType.ERROR.value
    
    return None

def map_haystack_finish_reason_to_finish_type(finish_reason):
    """Map Haystack finish_reason to standardized finish_type."""
    if not finish_reason:
        return None

    # Convert to lowercase for case-insensitive matching
    finish_reason_lower = finish_reason.lower() if isinstance(finish_reason, str) else str(finish_reason).lower()

    # Try direct mapping first
    if finish_reason in HAYSTACK_FINISH_REASON_MAPPING:
        return HAYSTACK_FINISH_REASON_MAPPING[finish_reason]

    # Try lowercase mapping
    if finish_reason_lower in HAYSTACK_FINISH_REASON_MAPPING:
        return HAYSTACK_FINISH_REASON_MAPPING[finish_reason_lower]

    # If no direct mapping, try to infer from common patterns
    if any(keyword in finish_reason_lower for keyword in ['stop', 'complete', 'success', 'done']):
        return FinishType.SUCCESS.value
    elif any(keyword in finish_reason_lower for keyword in ['length', 'token', 'limit', 'truncat']):
        return FinishType.TRUNCATED.value
    elif any(keyword in finish_reason_lower for keyword in ['filter', 'safety', 'block']):
        return FinishType.CONTENT_FILTER.value
    elif any(keyword in finish_reason_lower for keyword in ['error', 'fail', 'exception']):
        return FinishType.ERROR.value

    return None

def map_teamsai_finish_reason_to_finish_type(finish_reason):
    """Map TeamsAI finish_reason to standardized finish_type."""
    if not finish_reason:
        return None

    # Convert to lowercase for case-insensitive matching
    finish_reason_lower = finish_reason.lower() if isinstance(finish_reason, str) else str(finish_reason).lower()

    # Try direct mapping first
    if finish_reason in TEAMSAI_FINISH_REASON_MAPPING:
        return TEAMSAI_FINISH_REASON_MAPPING[finish_reason]

    # Try lowercase mapping
    if finish_reason_lower in TEAMSAI_FINISH_REASON_MAPPING:
        return TEAMSAI_FINISH_REASON_MAPPING[finish_reason_lower]

    return None

def map_adk_finish_reason_to_finish_type(finish_reason):
    """Map ADK finish_reason to standardized finish_type."""
    if not finish_reason:
        return None
    return ADK_FINISH_REASON_MAPPING.get(finish_reason, None)

def map_mistral_finish_reason_to_finish_type(finish_reason):
    """Map Mistral finish_reason to standardized finish_type."""
    if not finish_reason:
        return None
    return MISTRAL_FINISH_REASON_MAPPING.get(finish_reason, None)

def map_hf_finish_reason_to_finish_type(finish_reason):
    """Map Hugging Face finish_reason to standardized finish_type."""
    if not finish_reason:
        return None
    return HUGGING_FACE_FINISH_REASON_MAPPING.get(finish_reason, None)

def map_litellm_finish_reason_to_finish_type(finish_reason):
    """Map LiteLLM finish_reason to standardized finish_type."""
    if not finish_reason:
        return None
    return LITELLM_FINISH_REASON_MAPPING.get(finish_reason, None)