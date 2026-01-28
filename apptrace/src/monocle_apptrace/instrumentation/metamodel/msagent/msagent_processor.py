"""Processor handlers for Microsoft Agent Framework instrumentation."""

import logging
from opentelemetry.context import attach, set_value, detach, get_value
from monocle_apptrace.instrumentation.common.span_handler import SpanHandler
from monocle_apptrace.instrumentation.common.constants import AGENT_INVOCATION_SPAN_NAME, AGENT_NAME_KEY, AGENT_SESSION, INFERENCE_AGENT_DELEGATION, INFERENCE_TURN_END, LAST_AGENT_INVOCATION_ID, LAST_AGENT_NAME, SPAN_TYPES, INFERENCE_TOOL_CALL
from monocle_apptrace.instrumentation.common.utils import set_scope
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
    
    def skip_span(self, to_wrap, wrapped, instance, args, kwargs):
        """Skip ChatAgent.run span when it's the 2nd+ turn span (inside workflow/multi-agent)."""
        from monocle_apptrace.instrumentation.common.utils import is_scope_set
        
        # Skip if turn scope is already set - means we're the 2nd+ span
        # The first span (workflow or first agent) already created the turn
        if is_scope_set("agentic.turn"):
            logger.debug(f"Skipping ChatAgent.run span - turn scope already set (inside workflow/multi-agent)")
            return True
        return False
    
    def pre_tracing(self, to_wrap, wrapped, instance, args, kwargs):
        """Called before turn execution to extract and store agent information in context."""
        from monocle_apptrace.instrumentation.metamodel.msagent._helper import uses_chat_client, is_inside_workflow
        from monocle_apptrace.instrumentation.metamodel.msagent.entities.inference import AGENT, AGENT_REQUEST
        from monocle_apptrace.instrumentation.common.utils import is_scope_set
        from monocle_apptrace.instrumentation.common.scope_wrapper import start_scope
        
        # Store agent information in context for child spans to access
        agent_info = {}
        if hasattr(instance, "name"):
            agent_info["name"] = instance.name
        if hasattr(instance, "instructions"):
            agent_info["instructions"] = instance.instructions
        
        # Extract thread/session ID and set scope
        session_id_token = None
        thread = kwargs.get("thread")
        if thread is not None:
            # Extract thread ID from thread object
            thread_id = None
            if hasattr(thread, "service_thread_id"):
                thread_id = thread.service_thread_id
            elif hasattr(thread, "id"):
                thread_id = thread.id
            elif hasattr(thread, "thread_id"):
                thread_id = thread.thread_id
            
            if thread_id:
                session_id_token = set_scope(AGENT_SESSION, thread_id)
        
        # Attach agent info context
        context_token = None
        if agent_info:
            context_token = attach(set_value(MSAGENT_CONTEXT_KEY, agent_info))
        
        # Store both tokens for cleanup
        self._session_token = session_id_token
        self._context_token = context_token
        
        # Determine processor based on client type and context
        scope_name = AGENT_REQUEST.get("type")
        alternate_to_wrap = None
        
        if uses_chat_client(instance):
            # ChatClient: use recursive processor list to create turn + invocation
            logger.debug(f"ChatClient: setting output_processor_list=[AGENT_REQUEST, AGENT]")
            alternate_to_wrap = to_wrap.copy()
            alternate_to_wrap["output_processor_list"] = [AGENT_REQUEST, AGENT]
            # Clear output_processor if it exists to avoid confusion
            if "output_processor" in alternate_to_wrap:
                del alternate_to_wrap["output_processor"]
        else:
            # AssistantsClient: ChatAgent.run creates turn only, AssistantsClient methods create invocation
            if not is_scope_set(scope_name):
                # Create turn scope, AssistantsClient will create invocation later
                logger.debug(f"AssistantsClient: setting output_processor=AGENT_REQUEST")
                alternate_to_wrap = to_wrap.copy()
                alternate_to_wrap["output_processor"] = AGENT_REQUEST
        
        return context_token, alternate_to_wrap

    def post_task_processing(self, to_wrap, wrapped, instance, args, kwargs, result, ex, span: Span, parent_span: Span):
        self.hydrate_events(to_wrap, wrapped, instance, args, kwargs,
                            result, span=parent_span, is_post_exec=True)

class MSAgentAgentHandler(SpanHandler):
    """Handler for Microsoft Agent Framework agent invocations."""
    
    def skip_span(self, to_wrap, wrapped, instance, args, kwargs):
        """Skip get_response/get_streaming_response for ChatClient only in standalone mode."""
        from monocle_apptrace.instrumentation.common.utils import is_scope_set
        
        client_class = instance.__class__.__name__
        # For ChatClient: skip only in standalone mode (when no turn scope exists yet)
        # In workflow/multi-agent: turn scope exists, so get_response creates invocation
        if client_class == "AzureOpenAIChatClient":
            if is_scope_set("agentic.turn"):
                # Turn scope exists - we're in workflow/multi-agent, don't skip
                logger.debug(f"Not skipping get_response - turn scope exists (workflow), create invocation span")
                return False
            else:
                # No turn scope - standalone mode where ChatAgent.run creates both via processor_list
                logger.debug(f"Skipping get_response - standalone mode, ChatAgent.run creates via processor_list")
                return True
        # Don't skip for AssistantsClient - this is where invocation span is created
        return False
    
    def pre_tracing(self, to_wrap, wrapped, instance, args, kwargs):
        """Set agent name in context."""
        # Set agent name in context for propagation
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

    def pre_tracing(self, to_wrap, wrapped, instance, args, kwargs):
        """Set agent name and skip if appropriate."""
        from monocle_apptrace.instrumentation.common.utils import is_scope_set
        
        # For ChatClient: only create span if we're in recursive output_processor_list flow
        # This happens when ChatAgent.run uses output_processor_list=[AGENT_REQUEST, AGENT]
        # The second recursive call needs get_response to create the invocation span
        client_class = instance.__class__.__name__
        if client_class == "AzureOpenAIChatClient":
            # Skip only if invocation scope already exists (would be duplicate)
            if is_scope_set("agentic.invocation"):
                return None, None  # Skip span creation
        
        # Set agent name in context for propagation
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

    def skip_span(self, to_wrap, wrapped, instance, args, kwargs):
        """Check if span should be skipped based on pre_tracing result."""
        # If pre_tracing returned (None, None), skip the span
        return False  # Let pre_tracing handle the skip logic

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

