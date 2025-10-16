from opentelemetry.context import set_value, attach, detach, get_value
from monocle_apptrace.instrumentation.common.constants import AGENT_PREFIX_KEY, SCOPE_NAME
from monocle_apptrace.instrumentation.common.span_handler import SpanHandler
from monocle_apptrace.instrumentation.metamodel.langgraph._helper import (
   DELEGATION_NAME_PREFIX, get_name, is_root_agent_name, is_delegation_tool, LANGGRAPTH_AGENT_NAME_KEY
)
from monocle_apptrace.instrumentation.metamodel.langgraph.entities.inference import (
    AGENT_DELEGATION, AGENT_REQUEST
)
from monocle_apptrace.instrumentation.common.scope_wrapper import start_scope, stop_scope

class LanggraphAgentHandler(SpanHandler):
    def pre_tracing(self, to_wrap, wrapped, instance, args, kwargs):
        context = set_value(LANGGRAPTH_AGENT_NAME_KEY, get_name(instance))
        context = set_value(AGENT_PREFIX_KEY, DELEGATION_NAME_PREFIX, context)
        scope_name = AGENT_REQUEST.get("type")
        if scope_name is not None and is_root_agent_name(instance) and get_value(scope_name, context) is None:
            return start_scope(scope_name, scope_value=None, context=context)
        else:
            return attach(context)

    def post_tracing(self, to_wrap, wrapped, instance, args, kwargs, result, token):
        if token is not None:
            detach(token)

    # In multi agent scenarios, the root agent is the one that orchestrates the other agents. LangGraph generates an extra root level invoke()
    # call on top of the supervisor agent invoke().
    # This span handler resets the parent invoke call as generic type to avoid duplicate attributes/events in supervisor span and this root span.
    def hydrate_span(self, to_wrap, wrapped, instance, args, kwargs, result, span, parent_span = None, ex:Exception = None, is_post_exec:bool= False) -> bool:
        if is_root_agent_name(instance) and "parent.agent.span" in span.attributes:
            agent_request_wrapper = to_wrap.copy()
            agent_request_wrapper["output_processor"] = AGENT_REQUEST
        else:
            agent_request_wrapper = to_wrap
            if hasattr(instance, 'name') and parent_span is not None and not SpanHandler.is_root_span(parent_span):
                parent_span.set_attribute("parent.agent.span", True)
        return super().hydrate_span(agent_request_wrapper, wrapped, instance, args, kwargs, result, span, parent_span, ex, is_post_exec)

class LanggraphToolHandler(SpanHandler):
    # LangGraph uses an internal tool to initate delegation to other agents. The method is tool invoke() with tool name as `transfer_to_<agent_name>`.
    # Hence we usea different output processor for tool invoke() to format the span as agentic.delegation.
    def hydrate_span(self, to_wrap, wrapped, instance, args, kwargs, result, span, parent_span = None, ex:Exception = None, is_post_exec:bool= False) -> bool:
        if is_delegation_tool(instance):
            agent_request_wrapper = to_wrap.copy()
            agent_request_wrapper["output_processor"] = AGENT_DELEGATION
        else:
            agent_request_wrapper = to_wrap

        return super().hydrate_span(agent_request_wrapper, wrapped, instance, args, kwargs, result, span, parent_span, ex, is_post_exec)
    