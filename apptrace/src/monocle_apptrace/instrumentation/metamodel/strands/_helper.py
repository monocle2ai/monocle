
from monocle_apptrace.instrumentation.common.constants import SPAN_TYPES


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