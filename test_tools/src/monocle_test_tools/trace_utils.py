import inspect
import os
from opentelemetry.sdk.trace import Span

def get_input_from_span(span: Span) -> str:
    """
    Extracts the input text from the span attributes.
    
    Args:
        span (Span): The span object from which to extract the input.
        
    Returns:
        str: The extracted input text, or an empty string if not found.
    """
    for event in span.events:
        if event.name == "data.input":
            return event.attributes.get("input", "")
    return None

def get_output_from_span(span: Span) -> str:
    """
    Extracts the output text from the span attributes.

    Args:
        span (Span): The span object from which to extract the output.

    Returns:
        str: The extracted output text, or an empty string if not found.
    """
    for event in span.events:
        if event.name == "data.output":
            return event.attributes.get("response")
    return None

def get_agent_description_from_span(span: Span) -> str:
    """
    Extracts the agent description from the span attributes.

    Args:
        span (Span): The span object from which to extract the agent description.

    Returns:
        str: The extracted agent description, or an empty string if not found.
    """
    if span.attributes.get("span.type") == "agentic_invocation":
        return span.attributes.get("agent.description")
    return None

def get_function_signature(func, *args, **kwargs) -> str:
        # Create the call signature from func name + args/kwargs
        func_name = func.__name__
        
        # Convert args to string representations
        arg_strs = [repr(arg) for arg in args]
        
        # Convert kwargs to string representations
        kwarg_strs = [f"{k}={repr(v)}" for k, v in kwargs.items()]
        
        # Combine them
        all_args = ", ".join(arg_strs + kwarg_strs)
        call_signature = f"{func_name}({all_args})"
        return call_signature

def get_caller_file_line() -> str:
    try:
        # Get the previous frame in the stack, that is, the caller's frame
        caller_frame = inspect.currentframe().f_back.f_back
        # Extract the filename and line number relative to current working directory
        filename = os.path.relpath(caller_frame.f_code.co_filename, os.getcwd())
        line_number = caller_frame.f_lineno
        return f"{filename}#{line_number}: "
    except Exception as e:
        return ""

