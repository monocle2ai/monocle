from opentelemetry.context import get_value
from monocle_apptrace.instrumentation.common.utils import resolve_from_alias, get_json_dumps, extract_content_text
from monocle_apptrace.instrumentation.common.constants import AGENT_NAME_KEY, LAST_AGENT_INVOCATION_ID, LAST_AGENT_NAME
import logging
logger = logging.getLogger(__name__)

DELEGATION_NAME_PREFIX = 'transfer_to_'
ROOT_AGENT_NAME = 'LangGraph'
LANGGRAPTH_AGENT_NAME_KEY = "agent.langgraph"

def _last_message_content_from_state(state) -> str:
    """Last non-empty message content from a state dict, allowing custom '*messages' channels."""
    if not isinstance(state, dict):
        return ""
    candidate_keys = []
    if 'messages' in state:
        candidate_keys.append('messages')
    candidate_keys += [k for k in state
                       if k != 'messages' and isinstance(k, str) and k.endswith('messages')]
    candidate_keys += [k for k in state if k not in candidate_keys]
    for k in candidate_keys:
        v = state.get(k)
        if isinstance(v, list) and v:
            for msg in reversed(v):
                content = getattr(msg, 'content', None)
                if content:
                    return extract_content_text(content)
    return ""

def extract_agent_response(response):
    try:
        if response is None:
            return ""
        # stream processor result (SimpleNamespace with output_text from LanggraphStreamProcessor)
        output_text = getattr(response, 'output_text', None)
        if output_text:
            return str(output_text)
        # astream emits a checkpoint tuple ((), 'debug', {payload}) as its final item
        if isinstance(response, tuple) and len(response) >= 3 and isinstance(response[2], dict):
            values = response[2].get('payload', {}).get('values', {})
            return _last_message_content_from_state(values)
        # 2-tuple stream chunk (mode, data) from a nested subgraph astream
        if isinstance(response, tuple) and len(response) == 2 and isinstance(response[1], dict):
            return _last_message_content_from_state(response[1])
        if isinstance(response, dict):
            text = _last_message_content_from_state(response)
            if text:
                return text
            # node-keyed updates ({node: {state}})
            for value in response.values():
                if isinstance(value, dict):
                    text = _last_message_content_from_state(value)
                    if text:
                        return text
    except Exception as e:
        logger.warning("Warning: Error occurred in handle_response: %s", str(e))
    return ""

def agent_instructions(arguments):
    if callable(arguments['kwargs']['agent'].instructions):
        return arguments['kwargs']['agent'].instructions(arguments['kwargs']['context_variables'])
    else:
        return arguments['kwargs']['agent'].instructions

def is_single_agent_instance(instance) -> bool:
    if hasattr(instance, 'builder') and hasattr(instance.builder, 'nodes'):
        # Check for both old pattern ('agent' node) and new pattern ('model' node)
        return 'agent' in instance.builder.nodes or 'model' in instance.builder.nodes
    return False

def _human_messages_from_state(state) -> list:
    """Collect human/user message text across any '*messages' channel of a state dict."""
    out = []
    if not isinstance(state, dict):
        return out
    for k, v in state.items():
        if not (k == 'messages' or (isinstance(k, str) and k.endswith('messages'))):
            continue
        if not isinstance(v, list):
            continue
        for message in v:
            if hasattr(message, 'type') and message.type == "human":
                out.append(extract_content_text(message.content))
            elif isinstance(message, dict) and message.get('role') == "user" and 'content' in message:
                out.append(extract_content_text(message['content']))
            elif isinstance(message, dict) and message.get('type') == "human" and 'content' in message:
                out.append(extract_content_text(message['content']))
    return out

def extract_agent_input(arguments):
    try:
        input_obj = None
        if arguments.get('kwargs') and 'input' in arguments['kwargs']:
            input_obj = arguments['kwargs']['input']
        elif arguments.get('args') and len(arguments['args']) > 0 and isinstance(arguments['args'][0], dict):
            input_obj = arguments['args'][0]
        if isinstance(input_obj, dict):
            messages = _human_messages_from_state(input_obj)
            if messages:
                return get_json_dumps(messages)
            # No human message (nested subgraph): fall back to last message content
            text = _last_message_content_from_state(input_obj)
            if text:
                return get_json_dumps([text])
        return ""
    except Exception as e:
        logger.warning("Warning: Error occurred in extract_agent_input: %s", str(e))
        return ""

def extract_parent_command_message(ex):
    try:
        return ex.args[0].update
    except Exception as e:
        logger.debug("Warning: Error occurred in extract_parent_command_message: %s", str(e))
    return ""

def get_inference_endpoint(arguments):
    inference_endpoint = resolve_from_alias(arguments['instance'].client.__dict__, ['azure_endpoint', 'api_base', '_base_url'])
    return str(inference_endpoint)

def tools(instance):
    if hasattr(instance,'nodes') and ('tools' in instance.nodes):
        tools= instance.nodes['tools']
        if hasattr(tools,'bound') and hasattr(tools.bound,'tools_by_name'):
            return list(tools.bound.tools_by_name.keys())

def update_span_from_llm_response(response):
    meta_dict = {}
    token_usage = None
    if response is not None and "messages" in response and isinstance(response["messages"], list) and len(response["messages"]) > 0:
        token = response["messages"][-1]
        if token.response_metadata is not None:
            token_usage = token.response_metadata.get("token_usage")
        if token_usage is not None:
            meta_dict.update({"completion_tokens": token_usage.get('completion_tokens')})
            meta_dict.update({"prompt_tokens": token_usage.get('prompt_tokens')})
            meta_dict.update({"total_tokens": token_usage.get('total_tokens')})
    return meta_dict

def update_span_from_stream_response(response):
    meta_dict = {}
    try:
        usage = getattr(response, 'usage', None)
        if usage and isinstance(usage, dict):
            meta_dict.update({
                "completion_tokens": usage.get('completion_tokens', 0),
                "prompt_tokens": usage.get('prompt_tokens', 0),
                "total_tokens": usage.get('total_tokens', 0),
            })
    except Exception as e:
        logger.warning("Warning: Error in update_span_from_stream_response: %s", str(e))
    return meta_dict

def update_span_from_response(response):
    if response is not None and hasattr(response, 'usage'):
        return update_span_from_stream_response(response)
    return update_span_from_llm_response(response)

def extract_tool_response(result):
    if result is not None and hasattr(result, 'content'):
        return result.content
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        return str(result)
    if isinstance(result, list) and isinstance(result[0], str):
        return result[0]
    if isinstance(result, tuple) and len(result) > 1:
        if isinstance(result[1], dict) and 'structured_content' in result[1]:
            structured = result[1]['structured_content']
            if isinstance(structured, dict) and 'result' in structured:
                return str(structured['result'])
    return None

def get_status(result):
    if result is not None and hasattr(result, 'status'):
        return result.status
    return None

def extract_tool_input(arguments):
    if arguments['args'] and len(arguments['args']) > 0:
        tool_input = arguments['args'][0]
    else:
        tool_input:dict = arguments['kwargs'].copy()
        tool_input.pop('run_manager', None)  # remove run_manager if exists
        tool_input.pop('config', None)  # remove config if exists
    return str(tool_input)

    # if isinstance(tool_input, str):
    #     return [tool_input]
    # elif isinstance(tool_input, dict):
    #     # return array of key value pairs
    #     return [f"'{k}': '{str(v)}'" for k, v in tool_input.items()]
    # else:
    #     return [str(tool_input)]

def get_name(instance):
    return instance.name if hasattr(instance, 'name') else ""

def get_agent_name(instance) -> str:
    return get_name(instance)

def get_tool_type(span):
    if (span.attributes.get("is_mcp", False)):
        return "tool.mcp"
    else:
        return "tool.langgraph"

def get_tool_name(instance) -> str:
    return get_name(instance)

def is_delegation_tool(instance) -> bool:
    return get_name(instance).startswith(DELEGATION_NAME_PREFIX)

def get_target_agent(instance) -> str:
    return get_name(instance).replace(DELEGATION_NAME_PREFIX, '', 1)

def is_root_agent_name(instance) -> bool:
    return get_name(instance) == ROOT_AGENT_NAME

def get_source_agent() -> str:
    """Get the name of the agent that initiated the request."""
    from_agent = get_value(AGENT_NAME_KEY)
    return from_agent if from_agent is not None else ""

def get_description(instance) -> str:
    return instance.description if hasattr(instance, 'description') else ""

def get_agent_description(instance) -> str:
    """Get the description of the agent."""
    return get_description(instance)

def get_tool_description(instance) -> str:
    """Get the description of the tool."""
    return get_description(instance)

def extract_thread_id(args, kwargs) -> str:
    if 'config' not in kwargs and len(args) > 1:
        kwargs = {**kwargs, 'config': args[1]}
    thread_id = None
    if 'config' in kwargs and 'configurable' in kwargs['config']:
        thread_id = kwargs['config']['configurable'].get('thread_id')
    return thread_id