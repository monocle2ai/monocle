from opentelemetry.context import attach, detach, get_current, get_value, set_value
from monocle_apptrace.instrumentation.common.constants import AGENT_PREFIX_KEY, AGENT_SESSION
from monocle_apptrace.instrumentation.common.span_handler import SpanHandler
from monocle_apptrace.instrumentation.common.utils import set_scope
from monocle_apptrace.instrumentation.metamodel.llamaindex._helper import (
    LLAMAINDEX_AGENT_NAME_KEY,
    is_delegation_tool, 
    get_agent_name, get_name,
    set_current_agent, get_current_agent,
    get_current_agent_span_id,
    extract_session_id,
    set_from_agent_info,
    get_delegation_info,
    set_delegation_info,
    update_delegations_with_span_id
)

TOOL_INVOCATION_STARTED:str = "llamaindex.tool_invocation_started"

class DelegationHandler(SpanHandler):
    # LlamaIndex uses an internal tool to initate delegation to other agents. The method is tool invoke() with tool name as `transfer_to_<agent_name>`.
    # Hence we skip creating delegation spans for these internal delegation tools.
    def skip_span(self, to_wrap, wrapped, instance, args, kwargs):
        if is_delegation_tool(args, instance):
            return True
        return super().skip_span(to_wrap, wrapped, instance, args, kwargs)

# There are two different APIs for tool calling FunctionTool.call() and AgentWorkflow.tool_call(). In case of single agent calling tool, only the FunctionTool.call() is used. In case of multi agent case,
# the AgentWorkflow.tool_call() is used which inturn calls FunctionTool.call(). We can't entirely rely on the FunctionTool.call() to extract tool span details, especially the agent delegation details are not available there.
# Hence we want to distinguish between single agent tool call and multi agent tool call. In case of multi agent tool call, we suppress the FunctionTool.call() span and use AgentWorkflow.tool_call() span to capture the tool call details.
class LlamaIndexToolHandler(DelegationHandler):
    def pre_tracing(self, to_wrap, wrapped, instance, args, kwargs):
        cur_context = get_current()
        cur_context = set_value(TOOL_INVOCATION_STARTED, True, cur_context)
        current_agent = get_value(LLAMAINDEX_AGENT_NAME_KEY)
        if current_agent is not None:
            cur_context = set_value(LLAMAINDEX_AGENT_NAME_KEY, current_agent, cur_context)
        
        # Check if this is a handoff/delegation tool
        if is_delegation_tool(args, instance):
            
            # Extract target agent from args
            target_agent = None
            if len(args) > 2 and isinstance(args[2], dict) and 'to_agent' in args[2]:
                target_agent = args[2]['to_agent']
            elif 'to_agent' in kwargs:
                target_agent = kwargs['to_agent']
            
            if target_agent:
                # Use thread-local storage to get the current agent name
                source_agent = get_current_agent()
                
                if source_agent:
                    # Get the agent span_id from thread-local storage
                    source_agent_span_id = get_current_agent_span_id()
                    
                    # Store delegation info with the source agent's span_id
                    set_delegation_info(target_agent, source_agent, source_agent_span_id or "")
        
        return attach(cur_context), None

    def post_tracing(self, to_wrap, wrapped, instance, args, kwargs, return_value, token=None):
        if token:
            detach(token)

class LlamaIndexSingleAgenttToolHandlerWrapper(DelegationHandler):
    def skip_span(self, to_wrap, wrapped, instance, args, kwargs):
        if get_value(TOOL_INVOCATION_STARTED) == True:
            return True
        return super().skip_span(to_wrap, wrapped, instance, args, kwargs)

class LlamaIndexAgentHandler(SpanHandler):
    def pre_tracing(self, to_wrap, wrapped, instance, args, kwargs):
        cur_context = get_current()
        agent_name = get_name(instance)
        
        # For LlamaIndex multi-agent workflows, check delegation store first
        delegation_info = get_delegation_info(agent_name)
        
        # Store from_agent info in thread-local so entity accessors can retrieve it
        if delegation_info:
            set_from_agent_info(delegation_info['from_agent'], delegation_info['from_agent_span_id'])
        
        # Set both OpenTelemetry context and thread-local storage
        set_current_agent(agent_name)
        cur_context = set_value(LLAMAINDEX_AGENT_NAME_KEY, agent_name, cur_context)
        cur_context = set_value(AGENT_PREFIX_KEY, "handoff", cur_context)
        
        # Extract session id and add to context via set_scope BEFORE attaching
        session_id = extract_session_id(kwargs)
        if session_id:
            # set_scope will attach the context with baggage, so we pass our cur_context
            session_id_token = set_scope(AGENT_SESSION, session_id, cur_context)
            return session_id_token, None
        else:
            # No session, just attach our context directly
            context_token = attach(cur_context)
            return context_token, None

    # LlamaIndex uses direct OpenAI call for agent inferences. Given that the workflow type is set to llamaindex, the openAI inference does not record the input/output events.
    # To avoid this, we set the workflow type to generic for agent inference spans so we can capture the prompts and responses.
    def hydrate_span(self, to_wrap, wrapped, instance, args, kwargs, result, span, parent_span = None, ex:Exception = None, is_post_exec:bool= False) -> bool:
        retval = super().hydrate_span(to_wrap, wrapped, instance, args, kwargs, result, span, parent_span, ex, is_post_exec)
        
        # Update the delegation store with confirmed invocation span_id
        agent_name = get_name(instance)
        if span and span.attributes.get("span.type") == "agentic.invocation":
            if hasattr(span, 'context') and span.context and span.context.span_id:
                span_id = format(span.context.span_id, '016x')
                update_delegations_with_span_id(agent_name, span_id)
        
        if SpanHandler.is_root_span(parent_span):
            span.set_attribute(LLAMAINDEX_AGENT_NAME_KEY, "")
        else:
            parent_span.set_attribute(LLAMAINDEX_AGENT_NAME_KEY, get_agent_name(instance))
        return retval
