import logging
from monocle_apptrace.instrumentation.common.utils import MonocleSpanException, get_json_dumps, get_status_code
from monocle_apptrace.instrumentation.common.utils import (
    Option,
    get_keys_as_tuple,
    get_nested_value,
    try_option,
    get_exception_message,
    get_exception_status_code
)

logger = logging.getLogger(__name__)

def extract_messages(arguments):
    """
    Captures the input from Teams AI state.
    Args:
        arguments (dict): Arguments containing state and context information
    Returns:
        str: The input message or error message
    """
    try:
        # Get the memory object from kwargs
        kwargs = arguments.get("kwargs", {})
        messages = []

        # If memory exists, try to get the input from temp
        if "memory" in kwargs:
            memory = kwargs["memory"]
            # Check if it's a TurnState object
            if hasattr(memory, "get"):
                # Use proper TurnState.get() method
                temp = memory.get("temp")
                if temp and hasattr(temp, "get"):
                    input_value = temp.get("input")
                    if input_value:
                        messages.append({'user': str(input_value)})
        system_prompt = ""
        try:
            system_prompt = kwargs.get("template").prompt.sections[0].sections[0].template
            messages.append({'system': system_prompt})
        except Exception as e:
            print(f"Debug - Error accessing system prompt: {str(e)}")
            
        # Try alternative path through context if memory path fails
        context = kwargs.get("context")
        if hasattr(context, "activity") and hasattr(context.activity, "text"):
            messages.append({'user': str(context.activity.text)})

        return [get_json_dumps(message) for message in messages]
    except Exception as e:
        print(f"Debug - Arguments structure: {str(arguments)}")
        print(f"Debug - kwargs: {str(kwargs)}")
        if "memory" in kwargs:
            print(f"Debug - memory type: {type(kwargs['memory'])}")
        return f"Error capturing input: {str(e)}"

def capture_prompt_info(arguments):
    """Captures prompt information from ActionPlanner state"""
    try:
        kwargs = arguments.get("kwargs", {})
        prompt = kwargs.get("prompt")

        if isinstance(prompt, str):
            return prompt
        elif hasattr(prompt, "name"):
            return prompt.name

        return "No prompt information found"
    except Exception as e:
        return f"Error capturing prompt: {str(e)}"

def capture_prompt_template_info(arguments):
    """Captures prompt information from ActionPlanner state"""
    try:
        kwargs = arguments.get("kwargs", {})
        prompt = kwargs.get("prompt")

        if hasattr(prompt,"prompt") and prompt.prompt is not None:
            if "_text" in prompt.prompt.__dict__:
                prompt_template = prompt.prompt.__dict__.get("_text", None)
                return prompt_template
            elif "_sections" in prompt.prompt.__dict__ and prompt.prompt._sections is not None:
                sections = prompt.prompt._sections[0].__dict__.get("_sections", None)
                if sections is not None and "_template" in sections[0].__dict__:
                    return sections[0].__dict__.get("_template", None)


        return "No prompt information found"
    except Exception as e:
        return f"Error capturing prompt: {str(e)}"

def status_check(arguments):
    if hasattr(arguments["result"], "error") and arguments["result"].error is not None:
        error_msg:str = arguments["result"].error
        error_code:str = arguments["result"].status if hasattr(arguments["result"], "status") else "unknown"
        raise MonocleSpanException(f"Error: {error_code} - {error_msg}")

def get_prompt_template(arguments):
    pass
    return {
        "prompt_template_name": capture_prompt_info(arguments),
        "prompt_template": capture_prompt_template_info(arguments),
        "prompt_template_description": get_nested_value(arguments.get("kwargs", {}), ["prompt", "config", "description"]),
        "prompt_template_type": get_nested_value(arguments.get("kwargs", {}), ["prompt", "config", "type"])
    }

def get_status(arguments):
    if arguments["exception"] is not None:
        return 'error'
    elif get_status_code(arguments) == 'success':
        return 'success'
    else:
        return 'error'

def extract_assistant_message(arguments) -> str:
    status = get_status_code(arguments)
    messages = []
    role = "assistant"
    if status == 'success':
        if hasattr(arguments["result"], "message"):
            messages.append({role: arguments["result"].message.content})
        else:
            messages.append({role: str(arguments["result"])})
    else:
        if arguments["exception"] is not None:
            return get_exception_message(arguments)
        elif hasattr(arguments["result"], "error"):
            return arguments["result"].error
    return get_json_dumps(messages[0]) if messages else ""

def check_status(arguments):
    status = get_status_code(arguments)
    if status != 'success':
        raise MonocleSpanException(f"{status}")   

def extract_provider_name(instance):
    provider_url: Option[str] = try_option(getattr, instance._client.base_url, 'host')
    return provider_url.unwrap_or(None)


def extract_inference_endpoint(instance):
    inference_endpoint: Option[str] = try_option(getattr, instance._client, 'base_url').map(str)
    if inference_endpoint.is_none() and "meta" in instance.client.__dict__:
        inference_endpoint = try_option(getattr, instance.client.meta, 'endpoint_url').map(str)

    return inference_endpoint.unwrap_or(extract_provider_name(instance))