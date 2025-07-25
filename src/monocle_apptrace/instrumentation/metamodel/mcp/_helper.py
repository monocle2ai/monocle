from opentelemetry.context import get_value
from monocle_apptrace.instrumentation.common.utils import resolve_from_alias
import logging
import json

logger = logging.getLogger(__name__)

def log(arguments):
    print(f"Arguments: {arguments}")

def get_output_text(arguments):
    # arguments["result"].content[0].text
    if "result" in arguments and hasattr(arguments["result"], 'tools') and isinstance(arguments["result"].tools, list):
        tools = []
        for tool in arguments["result"].tools:
            if hasattr(tool, 'name'):
                tools.append(tool.name)
        return tools
    if "result" in arguments and hasattr(arguments["result"], 'content') and isinstance(arguments["result"].content, list):
        ret_val = []
        for content in arguments["result"].content:
            if hasattr(content, 'text'):
                ret_val.append(content.text)
        return ret_val

def get_name(arguments):
    """Get the name of the tool from the instance."""
    # args[0].root.params.name
    # json.dumps(args[0].root.params.arguments)
    
    args = arguments['args']
    if args and hasattr(args[0], 'root') and hasattr(args[0].root, 'params') and hasattr(args[0].root.params, 'name'):
        # If the first argument has a root with params and name, return that name
        return args[0].root.params.name
    
def get_type(arguments):
    """Get the type of the tool from the instance."""
    args = arguments['args']
    if args and hasattr(args[0], 'root') and hasattr(args[0].root, 'method'):
        # If the first argument has a root with a method, return that method's name
        return args[0].root.method

def get_params_arguments(arguments):
    """Get the params of the tool from the instance."""
    # args[0].root.params.name
    # json.dumps(args[0].root.params.arguments)
    
    args = arguments['args']
    if args and hasattr(args[0], 'root') and hasattr(args[0].root, 'params') and hasattr(args[0].root.params, 'arguments'):
        # If the first argument has a root with params and arguments, return those arguments
        try:
            return json.dumps(args[0].root.params.arguments)
        except (TypeError, ValueError) as e:
            logger.error(f"Error serializing arguments: {e}")
            return str(args[0].root.params.arguments)

