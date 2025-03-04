from monocle_apptrace.instrumentation.common.utils import resolve_from_alias
import logging
logger = logging.getLogger(__name__)

def handle_openai_response(response):
    try:
        if 'messages' in response:
            output = response["messages"][-1]
            return str(output.content)
    except Exception as e:
        logger.warning("Warning: Error occurred in handle_openai_response: %s", str(e))
        return ""

def agent_instructions(arguments):
    if callable(arguments['kwargs']['agent'].instructions):
        return arguments['kwargs']['agent'].instructions(arguments['kwargs']['context_variables'])
    else:
        return arguments['kwargs']['agent'].instructions

def extract_input(arguments):
    history = arguments['result']['messages']
    for message in history:
        if hasattr(message, 'content') and hasattr(message, 'type') and message.type == "human":  # Check if the message is a HumanMessage
            return message.content

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
    if response is not None and "messages" in response:
        token = response["messages"][-1]
        if token.response_metadata is not None:
            token_usage = token.response_metadata["token_usage"]
        if token_usage is not None:
            meta_dict.update({"completion_tokens": token_usage.get('completion_tokens')})
            meta_dict.update({"prompt_tokens": token_usage.get('prompt_tokens')})
            meta_dict.update({"total_tokens": token_usage.get('total_tokens')})
    return meta_dict
