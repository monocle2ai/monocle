"""Where an instruction goes in a model call, for each way a model SDK carries its system prompt.

Each function takes the call's keyword arguments and the instruction text. It returns new keyword
arguments with the instruction ahead of the developer's system prompt, or None if the instruction
cannot be added to this call. The developer's own objects are never changed.
"""
from typing import Optional


def add_system_message(kwargs: dict, text: str) -> Optional[dict]:
    """A chat call that takes ``messages``, directly or under ``data``: the instruction is its first system message."""
    if kwargs.get("messages"):
        return {**kwargs, "messages": placed_first({"role": "system", "content": text}, list(kwargs["messages"]))}
    if not isinstance(kwargs.get("data"), dict):
        return None
    instructed_data = add_system_message(kwargs["data"], text)
    if instructed_data is None:
        return None
    return {**kwargs, "data": instructed_data}


def add_to_instructions(kwargs: dict, text: str) -> Optional[dict]:
    """A Responses call: the instruction goes ahead of ``instructions``."""
    if "input" not in kwargs:
        return None
    instructions = instruction_before(text, kwargs.get("instructions"))
    if instructions is None:
        return None
    return {**kwargs, "instructions": instructions}


def add_to_messages_or_instructions(kwargs: dict, text: str) -> Optional[dict]:
    """An OpenAI call, which is either a chat call or a Responses call."""
    if "messages" in kwargs:
        return add_system_message(kwargs, text)
    return add_to_instructions(kwargs, text)


def add_to_request_instructions(kwargs: dict, text: str) -> Optional[dict]:
    """A LiteLLM Responses call: the instruction goes ahead of ``instructions`` in its request params."""
    request_params = kwargs.get("response_api_optional_request_params")
    if not isinstance(request_params, dict):
        return None
    instructions = instruction_before(text, request_params.get("instructions"))
    if instructions is None:
        return None
    return {**kwargs, "response_api_optional_request_params": {**request_params, "instructions": instructions}}


def add_to_system(kwargs: dict, text: str) -> Optional[dict]:
    """An Anthropic call: the instruction goes ahead of ``system``, whether that is text or blocks."""
    if "messages" not in kwargs:
        return None
    system = kwargs.get("system")
    if isinstance(system, (list, tuple)):
        return {**kwargs, "system": placed_first({"type": "text", "text": text}, list(system))}
    instructed_system = instruction_before(text, system)
    if instructed_system is None:
        return None
    return {**kwargs, "system": instructed_system}


def add_system_block(kwargs: dict, text: str) -> Optional[dict]:
    """A Bedrock Converse call: the instruction is the first ``system`` block.

    Returns None for other Bedrock calls. ``invoke_model`` is not supported: its body differs per model family.
    """
    if "messages" not in kwargs:
        return None
    return {**kwargs, "system": placed_first({"text": text}, list(kwargs.get("system") or []))}


def add_to_system_instruction(kwargs: dict, text: str) -> Optional[dict]:
    """A Gemini call: the instruction goes ahead of ``system_instruction`` in its config.

    Returns None if the call uses cached content, which cannot be combined with a system instruction,
    or if its system instruction is not text, parts or a Content.
    """
    if "contents" not in kwargs:
        return None
    config = kwargs.get("config")
    if gemini_config_value(config, "cached_content", "cachedContent"):
        return None
    system_instruction = gemini_system_instruction_with(
        text, gemini_config_value(config, "system_instruction", "systemInstruction"))
    if system_instruction is None:
        return None
    return {**kwargs, "config": gemini_config_with_system_instruction(config, system_instruction)}


def gemini_config_value(config, name: str, camel_case_name: str):
    if isinstance(config, dict):
        return config.get(name) or config.get(camel_case_name)
    return getattr(config, name, None)


def gemini_system_instruction_with(text: str, system_instruction):
    if isinstance(system_instruction, list):
        return placed_first(text, system_instruction)
    if hasattr(system_instruction, "parts"):
        return placed_first(text, list(system_instruction.parts or []))
    return instruction_before(text, system_instruction)


def gemini_config_with_system_instruction(config, system_instruction):
    if config is None:
        return {"system_instruction": system_instruction}
    if not isinstance(config, dict):
        return config.model_copy(update={"system_instruction": system_instruction})
    if "systemInstruction" in config:
        return {**config, "systemInstruction": system_instruction}
    return {**config, "system_instruction": system_instruction}


def placed_first(item, items: list) -> list:
    """The items with this one first, unless it already is first."""
    if items[:1] == [item]:
        return items
    return [item, *items]


def instruction_before(text: str, system_prompt) -> Optional[str]:
    """The instruction followed by the developer's system prompt.

    Returns None if there is a system prompt and it is not text, so it is never replaced.
    """
    if not system_prompt:
        return text
    if not isinstance(system_prompt, str):
        return None
    if system_prompt == text or system_prompt.startswith(f"{text}\n\n"):
        return system_prompt
    return f"{text}\n\n{system_prompt}"
