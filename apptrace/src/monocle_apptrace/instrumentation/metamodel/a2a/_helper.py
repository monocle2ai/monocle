from opentelemetry.context import get_value
from monocle_apptrace.instrumentation.common.utils import resolve_from_alias
import logging
import json

logger = logging.getLogger(__name__)

def get_url(arguments):
    """Get the URL of the agent being called.

    Older clients hold it themselves, newer ones on their transport.
    """
    instance = arguments["instance"]
    return getattr(instance, "url", None) or getattr(
        getattr(instance, "_transport", None), "url", None)

def get_method(arguments):
    """Get the method of the tool from the instance."""
    return arguments["args"][0].method

def _get_result(arguments):
    """Get the Task or Message the server answered with, or None.

    An error response carries neither. Accessor errors are swallowed by the
    span handler, so a miss here would drop the attribute silently.
    """
    return getattr(getattr(arguments.get("result"), "root", None), "result", None)

def _get_text_parts(parts):
    """Get the text of a list of parts, skipping file and data parts."""
    return [part.root.text for part in parts or []
            if getattr(part.root, "text", None)]

def get_params_arguments(arguments):
    """Get the params of the tool from the instance."""
    texts = _get_text_parts(arguments["args"][0].params.message.parts)
    return texts[0] if texts else None

def get_role(arguments):
    """Get the role of the tool from the instance."""
    return arguments["args"][0].params.message.role.value

def get_status(arguments):
    """Get the status of the tool from the result. Only a Task has one."""
    status = getattr(_get_result(arguments), "status", None)
    state = getattr(status, "state", None)
    return getattr(state, "value", None)

def get_response(arguments):
    """Get the response of the tool from the result.

    A Task answers through its artifacts, a Message through its own parts, and
    either may be missing. Returned as one string, like every other span's
    output, so an exact-match assertion reads what the agent said.
    """
    result = _get_result(arguments)
    if result is None:
        return None
    ret_val = []
    for artifact in getattr(result, "artifacts", None) or []:
        ret_val.extend(_get_text_parts(artifact.parts))
    if not ret_val:
        ret_val = _get_text_parts(getattr(result, "parts", None))
    return "\n".join(ret_val) if ret_val else None

