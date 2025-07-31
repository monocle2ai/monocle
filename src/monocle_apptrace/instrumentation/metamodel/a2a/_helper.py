from opentelemetry.context import get_value
from monocle_apptrace.instrumentation.common.utils import resolve_from_alias
import logging
import json

logger = logging.getLogger(__name__)

def get_url(arguments):
    """Get the URL of the tool from the instance."""
    return arguments["instance"].url

def get_method(arguments):
    """Get the method of the tool from the instance."""
    return arguments["args"][0].method

def get_params_arguments(arguments):
    """Get the params of the tool from the instance."""
    return arguments["args"][0].params.message.parts[0].root.text

def get_role(arguments):
    """Get the role of the tool from the instance."""
    return arguments["args"][0].params.message.role.value

def get_status(arguments):
    """Get the status of the tool from the result."""
    return arguments["result"].root.result.status.state.value

def get_response(arguments):
    """Get the response of the tool from the result."""
    ret_val = []
    for artifact in arguments["result"].root.result.artifacts:
        if artifact.parts:
            for part in artifact.parts:
                if part.root.text:
                    ret_val.append(part.root.text)
    return ret_val
    # return arguments["result"].root.result.artifacts[0].parts[0].root.text
