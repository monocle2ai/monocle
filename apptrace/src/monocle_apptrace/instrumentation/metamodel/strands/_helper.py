
from monocle_apptrace.instrumentation.common.constants import SPAN_TYPES

__all__ = [
    "extract_session_id",
    "get_agent_name",
    "get_agent_description",
    "extract_agent_input",
    "extract_agent_response",
    "get_tool_type",
    "get_tool_name",
    "get_tool_description",
    "get_source_agent",
    "extract_tool_input",
    "extract_tool_response",
    "should_skip_delegation",
    "should_skip_request",
]


def extract_session_id(instance):
    # AWS Strands manages sessions through a session_manager object
    # The session_id is typically accessible via instance.session_manager.session_id
    if hasattr(instance, 'session_manager') and instance.session_manager is not None:
        if hasattr(instance.session_manager, 'session_id'):
            return instance.session_manager.session_id
    return None


def get_agent_name(instance):
    return instance.name

def get_agent_description(instance):
    return instance.description

def extract_agent_input(arguments):
    return arguments['args'][0]

def extract_agent_response(result):
    return result.message['content'][0]['text']

def get_tool_type(arguments):
    ## TODO: check for MCP type
    return "tool.strands"

def get_tool_name(arguments):
    return arguments.get('args')[1][0]['name']

def get_tool_description(arguments):
    return arguments.get('args')[1][0]['description']

def get_source_agent(arguments):
    return arguments.get('args')[0].name

def extract_tool_input(arguments):
    return str(arguments.get('args')[1][0]['input'])

def extract_tool_response(result):
    return result.tool_result['content'][0]['text']

def should_skip_delegation(arguments):
    if arguments.get('parent_span') and arguments.get('parent_span').attributes.get("span.type") != SPAN_TYPES.AGENTIC_TOOL_INVOCATION:
        return True
    return False

def should_skip_request(arguments):
    if arguments.get('parent_span') and arguments.get('parent_span').attributes.get("span.type") in [SPAN_TYPES.AGENTIC_TOOL_INVOCATION, SPAN_TYPES.AGENTIC_REQUEST]:
        return True
    return False