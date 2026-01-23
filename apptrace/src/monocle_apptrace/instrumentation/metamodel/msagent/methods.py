"""Microsoft Agent Framework method definitions for instrumentation."""

# import inspect
from monocle_apptrace.instrumentation.common.wrapper import (
    atask_wrapper, 
    atask_iter_wrapper, 
    ascopes_wrapper,
    with_tracer_wrapper
)
from monocle_apptrace.instrumentation.common.utils import set_scopes, remove_scope
from opentelemetry.trace import Tracer
from monocle_apptrace.instrumentation.common.span_handler import SpanHandler
from monocle_apptrace.instrumentation.metamodel.msagent.entities.inference import (
    AGENT, 
    AGENT_REQUEST,
    INFERENCE,
    TOOL,
)

def extract_msteams_context(args, kwargs):
    """
    Extracts MS Teams context from Bot Framework TurnContext.
    This function looks for a TurnContext object in the arguments/kwargs
    and extracts Teams-specific metadata.
    """
    scopes: dict[str, str] = {}
    
    # Try to find TurnContext in various possible locations
    context = None
    
    # Check kwargs for common parameter names
    for param_name in ["context", "turn_context", "turnContext"]:
        if param_name in kwargs:
            context = kwargs[param_name]
            break
    
    # Check args if not found in kwargs
    if not context and len(args) > 0:
        for arg in args:
            # Check if this looks like a TurnContext object
            if hasattr(arg, 'activity') and hasattr(arg.activity, 'channel_id'):
                context = arg
                break
    
    if not context or not hasattr(context, 'activity'):
        return scopes
    
    activity = context.activity
    
    # Extract channel information
    if hasattr(activity, 'channel_id') and activity.channel_id:
        channel_id = activity.channel_id
        scopes["teams.channel.channel_id"] = channel_id
        
        # If it's MS Teams, extract Teams-specific attributes
        if channel_id == "msteams":
            # Activity type
            if hasattr(activity, 'type') and activity.type:
                scopes["msteams.activity.type"] = activity.type
            
            # Conversation information
            if hasattr(activity, "conversation") and activity.conversation:
                if hasattr(activity.conversation, 'id'):
                    scopes["msteams.conversation.id"] = activity.conversation.id or ""
                if hasattr(activity.conversation, 'conversation_type'):
                    scopes["msteams.conversation.type"] = activity.conversation.conversation_type or ""
                if hasattr(activity.conversation, 'name'):
                    scopes["msteams.conversation.name"] = activity.conversation.name or ""
            
            # User information (from_property)
            if hasattr(activity, "from_property") and activity.from_property:
                if hasattr(activity.from_property, 'id'):
                    scopes["msteams.user.from_property.id"] = activity.from_property.id or ""
                if hasattr(activity.from_property, 'name'):
                    scopes["msteams.user.from_property.name"] = activity.from_property.name or ""
                if hasattr(activity.from_property, 'role'):
                    scopes["msteams.user.from_property.role"] = activity.from_property.role or ""
            
            # Recipient information
            if hasattr(activity, "recipient") and activity.recipient:
                if hasattr(activity.recipient, 'id'):
                    scopes["msteams.recipient.id"] = activity.recipient.id or ""
            
            # Channel data (tenant, team, channel details)
            if hasattr(activity, "channel_data") and activity.channel_data:
                channel_data = activity.channel_data
                
                # Tenant information
                if isinstance(channel_data, dict):
                    if "tenant" in channel_data and "id" in channel_data["tenant"]:
                        scopes["msteams.channel_data.tenant.id"] = channel_data["tenant"]["id"] or ""
                    
                    # Team information
                    if "team" in channel_data:
                        if "id" in channel_data["team"]:
                            scopes["msteams.channel_data.team.id"] = channel_data["team"]["id"] or ""
                        if "name" in channel_data["team"]:
                            scopes["msteams.channel_data.team.name"] = channel_data["team"]["name"] or ""
                    
                    # Channel information
                    if "channel" in channel_data:
                        if "id" in channel_data["channel"]:
                            scopes["msteams.channel_data.channel.id"] = channel_data["channel"]["id"] or ""
                        if "name" in channel_data["channel"]:
                            scopes["msteams.channel_data.channel.name"] = channel_data["channel"]["name"] or ""
    
    return scopes


@with_tracer_wrapper
async def msteams_context_wrapper(tracer: Tracer, handler: SpanHandler, to_wrap, wrapped, instance, source_path, args, kwargs):
    """
    Custom wrapper for MS Teams context extraction.
    Extracts Teams context from TurnContext, sets it as scopes, 
    then removes the context parameter before calling the wrapped function.
    """
    # Extract scope values using the extraction function
    scope_values = extract_msteams_context(args, kwargs)
    token = None
    
    # Create a filtered kwargs dict without the context parameters
    filtered_kwargs = {k: v for k, v in kwargs.items() 
                       if k not in ["context", "turn_context", "turnContext"]}
    
    try:
        if scope_values:
            token = set_scopes(scope_values)
        # Call wrapped function with filtered kwargs (without context parameter)
        return_value = await wrapped(*args, **filtered_kwargs)
        return return_value
    finally:
        if token:
            remove_scope(token)


MSAGENT_METHODS = [
    # Workflow-level methods - top level span for multi-agent workflows
    {
        "package": "agent_framework._workflows._handoff",
        "object": "Workflow",
        "method": "run",
        "span_handler": "msagent_request_handler",
        "wrapper_method": atask_wrapper,
        "output_processor": AGENT_REQUEST,
    },
    {
        "package": "agent_framework",
        "object": "ChatAgent",
        "method": "run_stream",
        "span_handler": "msagent_request_handler",
        "wrapper_method": atask_iter_wrapper,
        "output_processor": AGENT_REQUEST,
    },
    {
        "package": "agent_framework",
        "object": "ChatAgent",
        "method": "run",
        "span_handler": "msagent_request_handler",
        "wrapper_method": atask_wrapper,
        "output_processor": AGENT_REQUEST,
    },
    {
        "package": "agent_framework.azure._chat_client",
        "object": "AzureOpenAIChatClient",
        "method": "get_streaming_response",
        "span_handler": "msagent_agent_handler",
        "wrapper_method": atask_iter_wrapper,
        "output_processor": AGENT,
    },
    {
        "package": "agent_framework.azure._chat_client",
        "object": "AzureOpenAIChatClient",
        "method": "get_response",
        "span_handler": "msagent_agent_handler",
        "wrapper_method": atask_wrapper,
        "output_processor": AGENT,
    },
    {
        "package": "agent_framework.azure._assistants_client",
        "object": "AzureOpenAIAssistantsClient",
        "method": "get_streaming_response",
        "span_handler": "msagent_agent_handler",
        "wrapper_method": atask_iter_wrapper,
        "output_processor": AGENT,
    },
    {
        "package": "agent_framework.azure._assistants_client",
        "object": "AzureOpenAIAssistantsClient",
        "method": "get_response",
        "span_handler": "msagent_agent_handler",
        "wrapper_method": atask_wrapper,
        "output_processor": AGENT,
    },
    {
        "package": "agent_framework._tools",
        "object": "AIFunction",
        "method": "invoke",
        "wrapper_method": atask_wrapper,
        "output_processor": TOOL,
    },
        {
        "package": "agent_framework.azure._assistants_client",
        "object": "AzureOpenAIAssistantsClient",
        "method": "_inner_get_response",
        "span_handler": "msagent_inference_handler",
        "wrapper_method": atask_wrapper,
        "output_processor": INFERENCE,
    },
    # MS Teams context extraction for ChatAgent methods
    # Uses custom wrapper that filters out context parameter after extraction
    {
        "package": "agent_framework",
        "object": "ChatAgent",
        "method": "run",
        "wrapper_method": msteams_context_wrapper,
    },
    {
        "package": "agent_framework",
        "object": "ChatAgent",
        "method": "run_stream",
        "wrapper_method": msteams_context_wrapper,
    },

]
