"""Processor handlers for Microsoft Agent Framework instrumentation."""

import logging
from opentelemetry.context import attach, set_value, detach
from monocle_apptrace.instrumentation.common.span_handler import SpanHandler

logger = logging.getLogger(__name__)

# Context key for storing agent information
MSAGENT_CONTEXT_KEY = "msagent.agent_info"


class MSAgentRequestHandler(SpanHandler):
    """Handler for Microsoft Agent Framework turn-level requests (agentic.request)."""

    def pre_tracing(self, to_wrap, wrapped, instance, args, kwargs):
        """Called before turn execution to extract and store agent information in context."""
        print(f"🎯 MSAgent Request Handler: pre_tracing called for {instance.__class__.__name__}")
        print(f"    Method: {to_wrap.get('method')}, Package: {to_wrap.get('package')}")
        print(f"    Args: {args}, Kwargs: {kwargs}")
        
        # Store agent information in context for child spans to access
        agent_info = {}
        if hasattr(instance, "name"):
            agent_info["name"] = instance.name
            print(f"    Storing agent name in context: {instance.name}")
        if hasattr(instance, "instructions"):
            agent_info["instructions"] = instance.instructions
        
        if agent_info:
            # Attach the context so it's available to child spans
            token = attach(set_value(MSAGENT_CONTEXT_KEY, agent_info))
            print(f"    Context attached with token: {token}")
            return token, None
        
        return None, None

    def post_tracing(self, to_wrap, wrapped, instance, args, kwargs, result, token):
        """Called after turn execution to clean up context."""
        print(f"🎯 MSAgent Request Handler: post_tracing called for {instance.__class__.__name__}")
        print(f"    Result: {result}, Token: {token}")
        if token is not None:
            print(f"    Detaching context token: {token}")
            detach(token)

    def hydrate_span(self, to_wrap, wrapped, instance, args, kwargs, result, span, parent_span=None, ex: Exception = None, is_post_exec: bool = False) -> bool:
        """Hydrate span with request-specific attributes."""
        print(f"🎯 MSAgent Request Handler: hydrate_span called for {instance.__class__.__name__}")
        print(f"    Span: {span.name if span else 'None'}, Is_post_exec: {is_post_exec}")
        return super().hydrate_span(to_wrap, wrapped, instance, args, kwargs, result, span, parent_span, ex, is_post_exec)


class MSAgentAgentHandler(SpanHandler):
    """Handler for Microsoft Agent Framework agent invocations."""

    def pre_tracing(self, to_wrap, wrapped, instance, args, kwargs):
        """Called before agent execution to extract agent information."""
        print(f"🤖 MSAgent Agent Handler: pre_tracing called for {instance.__class__.__name__}")
        print(f"    Method: {to_wrap.get('method')}, Package: {to_wrap.get('package')}")
        print(f"    Args: {args}, Kwargs: {kwargs}")
        return None, None

    def post_tracing(self, to_wrap, wrapped, instance, args, kwargs, result, token):
        """Called after agent execution to extract result information."""
        print(f"🤖 MSAgent Agent Handler: post_tracing called for {instance.__class__.__name__}")
        print(f"    Result: {result}")
        if token is not None:
            detach(token)

    def hydrate_span(self, to_wrap, wrapped, instance, args, kwargs, result, span, parent_span=None, ex: Exception = None, is_post_exec: bool = False) -> bool:
        """Hydrate span with agent-specific attributes."""
        print(f"🤖 MSAgent Agent Handler: hydrate_span called for {instance.__class__.__name__}")
        print(f"    Span: {span.name if span else 'None'}, Is_post_exec: {is_post_exec}")
        return super().hydrate_span(to_wrap, wrapped, instance, args, kwargs, result, span, parent_span, ex, is_post_exec)


class MSAgentToolHandler(SpanHandler):
    """Handler for Microsoft Agent Framework tool invocations."""

    def pre_tracing(self, to_wrap, wrapped, instance, args, kwargs):
        """Called before tool execution to extract tool information."""
        print(f"🔧 MSAgent Tool Handler: pre_tracing called for {instance.__class__.__name__}")
        print(f"    Method: {to_wrap.get('method')}, Package: {to_wrap.get('package')}")
        print(f"    Args: {args}, Kwargs: {kwargs}")
        return None, None

    def post_tracing(self, to_wrap, wrapped, instance, args, kwargs, result, token):
        """Called after tool execution to extract result information."""
        print(f"🔧 MSAgent Tool Handler: post_tracing called for {instance.__class__.__name__}")
        print(f"    Result: {result}")
        if token is not None:
            detach(token)

    def hydrate_span(self, to_wrap, wrapped, instance, args, kwargs, result, span, parent_span=None, ex: Exception = None, is_post_exec: bool = False) -> bool:
        """Hydrate span with tool-specific attributes."""
        print(f"🔧 MSAgent Tool Handler: hydrate_span called for {instance.__class__.__name__}")
        print(f"    Span: {span.name if span else 'None'}, Is_post_exec: {is_post_exec}")
        return super().hydrate_span(to_wrap, wrapped, instance, args, kwargs, result, span, parent_span, ex, is_post_exec)
