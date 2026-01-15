"""Processor handlers for Microsoft Agent Framework instrumentation."""

import logging
from opentelemetry.context import attach, set_value, detach, get_value
from monocle_apptrace.instrumentation.common.span_handler import SpanHandler
from monocle_apptrace.instrumentation.common.constants import AGENT_INVOCATION_SPAN_NAME, AGENT_NAME_KEY, INFERENCE_AGENT_DELEGATION, INFERENCE_TURN_END, LAST_AGENT_INVOCATION_ID, LAST_AGENT_NAME, SPAN_TYPES, INFERENCE_TOOL_CALL
from opentelemetry.trace import Span

logger = logging.getLogger(__name__)

# Context key for storing agent information
MSAGENT_CONTEXT_KEY = "msagent.agent_info"

def propogate_agent_name_to_parent_span(span: Span, parent_span: Span):
    """Propagate agent name from child span to parent span."""
    if span.attributes.get("span.type") != AGENT_INVOCATION_SPAN_NAME:
        return
    if parent_span is not None:
        parent_span.set_attribute(LAST_AGENT_INVOCATION_ID, hex(span.context.span_id))
        # Try to get agent name from context first, then fall back to span attributes
        agent_name = get_value(AGENT_NAME_KEY)
        if agent_name is None:
            # Context may have been detached, try reading from span attributes
            agent_name = span.attributes.get("entity.1.name")
        if agent_name is not None:
            parent_span.set_attribute(LAST_AGENT_NAME, agent_name)

class MSAgentRequestHandler(SpanHandler):
    """Handler for Microsoft Agent Framework turn-level requests (agentic.request)."""
    def pre_tracing(self, to_wrap, wrapped, instance, args, kwargs):
        """Called before turn execution to extract and store agent information in context."""
        # Store agent information in context for child spans to access
        agent_info = {}
        if hasattr(instance, "name"):
            agent_info["name"] = instance.name
        if hasattr(instance, "instructions"):
            agent_info["instructions"] = instance.instructions
        if agent_info:
            # Attach the context so it's available to child spans
            token = attach(set_value(MSAGENT_CONTEXT_KEY, agent_info))
            return token, None
        
        return None, None

class MSAgentAgentHandler(SpanHandler):
    """Handler for Microsoft Agent Framework agent invocations."""

    def pre_tracing(self, to_wrap, wrapped, instance, args, kwargs):
        """Called before agent execution to set agent name in context."""
        # Set agent name in context for propagation to parent span
        agent_name = None
        if hasattr(instance, "name"):
            agent_name = instance.name
        elif hasattr(instance, "_name"):
            agent_name = instance._name
        if agent_name:
            context = set_value(AGENT_NAME_KEY, agent_name)
            token = attach(context)
            return token, None
        return None, None

    def post_tracing(self, to_wrap, wrapped, instance, args, kwargs, result, token):
        """Called after agent execution to clean up context."""
        self._context_token = token  # Store for later cleanup

    def post_task_processing(self, to_wrap, wrapped, instance, args, kwargs, result, ex, span, parent_span):
        """Propagate agent name and invocation ID to parent span, then clean up context."""
        # Propagate while context still has agent name
        propogate_agent_name_to_parent_span(span, parent_span)
        # Now detach context
        if hasattr(self, '_context_token') and self._context_token is not None:
            detach(self._context_token)
            self._context_token = None
        return super().post_task_processing(to_wrap, wrapped, instance, args, kwargs, result, ex, span, parent_span)

    def hydrate_span(self, to_wrap, wrapped, instance, args, kwargs, result, span, parent_span=None, ex: Exception = None, is_post_exec: bool = False) -> bool:
        """Hydrate span with agent-specific attributes."""
        return super().hydrate_span(to_wrap, wrapped, instance, args, kwargs, result, span, parent_span, ex, is_post_exec)


class MSAgentToolHandler(SpanHandler):
    """Handler for Microsoft Agent Framework tool invocations."""

    def pre_tracing(self, to_wrap, wrapped, instance, args, kwargs):
        """Called before tool execution to extract tool information."""
        return None, None

    def post_tracing(self, to_wrap, wrapped, instance, args, kwargs, result, token):
        """Called after tool execution to extract result information."""
        if token is not None:
            detach(token)

    def hydrate_span(self, to_wrap, wrapped, instance, args, kwargs, result, span, parent_span=None, ex: Exception = None, is_post_exec: bool = False) -> bool:
        """Hydrate span with tool-specific attributes."""
        return super().hydrate_span(to_wrap, wrapped, instance, args, kwargs, result, span, parent_span, ex, is_post_exec)

class MSAgentInferenceHandler(SpanHandler):
    """Handler for OpenAI inference spans in Microsoft Agent Framework context.
    
    This handler modifies inference span subtypes for MS Agent Framework:
    - Handoffs (delegations like 'handoff_to_*') should show as 'turn_end' 
      because no actual tool is invoked - the agent just transfers control.
    - Actual tool calls (book_flight, book_hotel, etc.) should stay as 'tool_call'
      because they invoke real tools with agentic.tool.invocation spans.
    """

    def hydrate_span(self, to_wrap, wrapped, instance, args, kwargs, result, span, parent_span=None, ex: Exception = None, is_post_exec: bool = False) -> bool:
        """Override inference span subtype for handoff scenarios."""
        # Only process in post-execution phase when span is complete
        if not is_post_exec or span is None:
            return super().hydrate_span(to_wrap, wrapped, instance, args, kwargs, result, span, parent_span, ex, is_post_exec)
        
        # Check if this is an inference span with agent delegation subtype
        span_type = span.attributes.get("span.type")
        span_subtype = span.attributes.get("span.subtype")
        tool_name = span.attributes.get("entity.3.name", "")
        
        if span_type == SPAN_TYPES.INFERENCE and span_subtype == INFERENCE_AGENT_DELEGATION:
            # Check if this is a handoff (starts with 'handoff_to') or an actual tool call
            if tool_name and tool_name.startswith("handoff_to"):
                # This is a handoff - no actual tool invocation, just control transfer
                span.set_attribute("span.subtype", INFERENCE_TURN_END)
            else:
                # This is an actual tool call - should stay as tool_call, not agent_delegation
                span.set_attribute("span.subtype", INFERENCE_TOOL_CALL)
        
        return super().hydrate_span(to_wrap, wrapped, instance, args, kwargs, result, span, parent_span, ex, is_post_exec)