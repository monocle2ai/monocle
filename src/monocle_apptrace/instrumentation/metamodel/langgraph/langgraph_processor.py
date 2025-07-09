from monocle_apptrace.instrumentation.common.span_handler import SpanHandler
from monocle_apptrace.instrumentation.metamodel.langgraph._helper import (
   get_name, is_root_agent_name, is_delegation_tool
)
from monocle_apptrace.instrumentation.metamodel.langgraph.entities.inference import (
     AGENT_GENERIC, AGENT_DELEGATION
)

class LanggraphAgentHandler(SpanHandler):
    # In multi agent scenarios, the root agent is the one that orchestrates the other agents. LangGraph generates an extra root level invoke()
    # call on top of the supervisor agent invoke().
    # This span handler resets the parent invoke call as generic type to avoid duplicate attributes/events in supervisor span and this root span.
    def hydrate_span(self, to_wrap, wrapped, instance, args, kwargs, result, span, parent_span = None, ex:Exception = None) -> bool:
        if is_root_agent_name(instance) and "parent.agent.span" in span.attributes:
            agent_request_wrapper = to_wrap.copy()
            agent_request_wrapper["output_processor"] = AGENT_GENERIC
        else:
            agent_request_wrapper = to_wrap
            if hasattr(instance, 'name') and parent_span is not None and not SpanHandler.is_root_span(parent_span):
                parent_span.set_attribute("parent.agent.span", True)
        return super().hydrate_span(agent_request_wrapper, wrapped, instance, args, kwargs, result, span, parent_span, ex)

class LanggraphToolHandler(SpanHandler):
    # LangGraph uses an internal tool to initate delegation to other agents. The method is tool invoke() with tool name as `transfer_to_<agent_name>`.
    # Hence we usea different output processor for tool invoke() to format the span as agentic.delegation.
    def hydrate_span(self, to_wrap, wrapped, instance, args, kwargs, result, span, parent_span = None, ex:Exception = None) -> bool:
        if is_delegation_tool(instance):
            agent_request_wrapper = to_wrap.copy()
            agent_request_wrapper["output_processor"] = AGENT_DELEGATION
        else:
            agent_request_wrapper = to_wrap

        return super().hydrate_span(agent_request_wrapper, wrapped, instance, args, kwargs, result, span, parent_span, ex)
    