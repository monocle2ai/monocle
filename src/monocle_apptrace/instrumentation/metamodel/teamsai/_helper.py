from monocle_apptrace.instrumentation.common.utils import MonocleSpanException
from monocle_apptrace.instrumentation.common.utils import (
    Option,
    get_keys_as_tuple,
    get_nested_value,
    try_option,
)
def capture_input(arguments):
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
                        return str(input_value)

        # Try alternative path through context if memory path fails
        context = kwargs.get("context")
        if hasattr(context, "activity") and hasattr(context.activity, "text"):
            return str(context.activity.text)

        return "No input found in memory or context"
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

def status_check(arguments):
    if hasattr(arguments["result"], "error") and arguments["result"].error is not None:
        error_msg:str = arguments["result"].error
        error_code:str = arguments["result"].status if hasattr(arguments["result"], "status") else "unknown"
        raise MonocleSpanException(f"Error: {error_code} - {error_msg}")

def extract_provider_name(instance):
    provider_url: Option[str] = try_option(getattr, instance._client.base_url, 'host')
    return provider_url.unwrap_or(None)


def extract_inference_endpoint(instance):
    inference_endpoint: Option[str] = try_option(getattr, instance._client, 'base_url').map(str)
    if inference_endpoint.is_none() and "meta" in instance.client.__dict__:
        inference_endpoint = try_option(getattr, instance.client.meta, 'endpoint_url').map(str)

    return inference_endpoint.unwrap_or(extract_provider_name(instance))