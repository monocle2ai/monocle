from opentelemetry.sdk.trace import Span, ReadableSpan

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