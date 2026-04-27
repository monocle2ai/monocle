"""
Custom span processor for capturing function inputs and outputs.
"""
import logging
import json
from monocle_apptrace.instrumentation.common.utils import get_error_message

logger = logging.getLogger(__name__)


def serialize_value(value, max_depth=3, current_depth=0):
    """
    Safely serialize values for span events.
    Handles various data types and prevents infinite recursion.
    
    Args:
        value: The value to serialize
        max_depth: Maximum recursion depth
        current_depth: Current depth in recursion
        
    Returns:
        Serialized value suitable for JSON
    """
    if current_depth >= max_depth:
        return str(value)
    
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    elif isinstance(value, (list, tuple)):
        return [serialize_value(v, max_depth, current_depth + 1) for v in value[:100]]  # Limit to 100 items
    elif isinstance(value, dict):
        return {k: serialize_value(v, max_depth, current_depth + 1) for k, v in list(value.items())[:100]}
    elif hasattr(value, '__dict__'):
        try:
            return serialize_value(value.__dict__, max_depth, current_depth + 1)
        except Exception:
            return str(value)
    else:
        return str(value)


def extract_input(arguments):
    """Extract and serialize function input arguments."""
    try:
        args = arguments.get("args", ())
        kwargs = arguments.get("kwargs", {})
        
        input_data = {
            "args": [serialize_value(arg) for arg in args] if args else [],
            "kwargs": {k: serialize_value(v) for k, v in kwargs.items()} if kwargs else {}
        }
        
        # Convert to JSON string for storage
        return json.dumps(input_data)
    except Exception as e:
        logger.warning(f"Failed to extract custom span input: {e}")
        return json.dumps({"error": "Failed to capture input"})


def extract_output(arguments):
    """Extract and serialize function output."""
    try:
        result = arguments.get("result")
        
        # Handle errors
        error_msg = get_error_message(arguments)
        if error_msg:
            return json.dumps({"error": error_msg})
        
        # Serialize the result
        output_data = {
            "result": serialize_value(result)
        }
        
        return json.dumps(output_data)
    except Exception as e:
        logger.warning(f"Failed to extract custom span output: {e}")
        return json.dumps({"error": "Failed to capture output"})


# Custom span entity processor following Monocle's pattern
CUSTOM_SPAN_PROCESSOR = {
    "type": "custom",
    "events": [
        {
            "name": "data.input",
            "attributes": [
                {
                    "_comment": "Captured function input (args and kwargs)",
                    "attribute": "input",
                    "accessor": lambda arguments: extract_input(arguments)
                }
            ]
        },
        {
            "name": "data.output",
            "attributes": [
                {
                    "_comment": "Error code if any",
                    "attribute": "error_code",
                    "accessor": lambda arguments: get_error_message(arguments)
                },
                {
                    "_comment": "Captured function output",
                    "attribute": "response",
                    "accessor": lambda arguments: extract_output(arguments)
                }
            ]
        }
    ]
}
