from opentelemetry.context import attach, detach, get_current, get_value, set_value, Context
from monocle_apptrace.instrumentation.common.constants import AGENT_PREFIX_KEY
from monocle_apptrace.instrumentation.common.span_handler import SpanHandler
from monocle_apptrace.instrumentation.metamodel.llamaindex._helper import (
    is_delegation_tool, LLAMAINDEX_AGENT_NAME_KEY, get_agent_name, get_name, set_current_agent
)
from monocle_apptrace.instrumentation.metamodel.llamaindex.entities.agent import (
    AGENT_DELEGATION
)

TOOL_INVOCATION_STARTED:str = "llamaindex.tool_invocation_started"

class DelegationHandler(SpanHandler):
    # LlamaIndex uses an internal tool to initate delegation to other agents. The method is tool invoke() with tool name as `transfer_to_<agent_name>`.
    # Hence we usea different output processor for tool invoke() to format the span as agentic.delegation.
    def hydrate_span(self, to_wrap, wrapped, instance, args, kwargs, result, span, parent_span = None, ex:Exception = None) -> bool:
        if is_delegation_tool(args, instance):
            agent_request_wrapper = to_wrap.copy()
            agent_request_wrapper["output_processor"] = AGENT_DELEGATION
        else:
            agent_request_wrapper = to_wrap

        return super().hydrate_span(agent_request_wrapper, wrapped, instance, args, kwargs, result, span, parent_span, ex)

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
        return attach(cur_context)

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

        # Set both OpenTelemetry context and thread-local storage
        set_current_agent(agent_name)
        cur_context = set_value(LLAMAINDEX_AGENT_NAME_KEY, agent_name, cur_context)
        cur_context = set_value(AGENT_PREFIX_KEY, "handoff", cur_context)
        return attach(cur_context)

    def post_tracing(self, to_wrap, wrapped, instance, args, kwargs, return_value, token=None):
        if token:
            detach(token)
    
    # LlamaIndex uses direct OpenAI call for agent inferences. Given that the workflow type is set to llamaindex, the openAI inference does not record the input/output events.
    # To avoid this, we set the workflow type to generic for agent inference spans so we can capture the prompts and responses.
    def hydrate_span(self, to_wrap, wrapped, instance, args, kwargs, result, span, parent_span = None, ex:Exception = None) -> bool:
        retval = super().hydrate_span(to_wrap, wrapped, instance, args, kwargs, result, span, parent_span, ex)
        if SpanHandler.is_root_span(parent_span):
            span.set_attribute(LLAMAINDEX_AGENT_NAME_KEY, "")
        else:
            parent_span.set_attribute(LLAMAINDEX_AGENT_NAME_KEY, get_agent_name(instance))
        return retval
