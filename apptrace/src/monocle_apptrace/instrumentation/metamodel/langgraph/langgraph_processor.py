import logging
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
try:
    from langgraph.errors import ParentCommand
except ImportError:
    ParentCommand = None

logger = logging.getLogger(__name__)

class ParentCommandFilterSpan:
    """A wrapper for spans that filters out ParentCommand exceptions from being recorded."""
    
    def __init__(self, span):
        self.span = span
        self.original_record_exception = span.record_exception
        
    def record_exception(self, exception, attributes=None, timestamp=None, escaped=False):
        """Filter out ParentCommand exceptions before recording them."""
        try:
            # Check if this is a ParentCommand exception
            if ParentCommand is not None and isinstance(exception, ParentCommand):
                logger.debug("Filtering out ParentCommand exception from span recording")
                return  # Don't record ParentCommand exceptions
            
            # For all other exceptions, use the original record_exception method
            return self.original_record_exception(exception, attributes, timestamp, escaped)
        except Exception as e:
            logger.debug(f"Error in ParentCommand filtering: {e}")
            # If filtering fails, fall back to original behavior
            return self.original_record_exception(exception, attributes, timestamp, escaped)

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

    def post_task_processing(self, to_wrap, wrapped, instance, args, kwargs, result, ex, span, parent_span):
        """Apply ParentCommand filtering to the span before task execution."""
        # Apply ParentCommand filtering to this span
        self._apply_parent_command_filtering(span)
        super().post_task_processing(to_wrap, wrapped, instance, args, kwargs, result, ex, span, parent_span)

    def _apply_parent_command_filtering(self, span):
        """Apply ParentCommand exception filtering to a span."""
        try:
            if hasattr(span, 'record_exception'):
                # Create a filtered wrapper and replace the record_exception method
                filter_wrapper = ParentCommandFilterSpan(span)
                span.record_exception = filter_wrapper.record_exception
                logger.debug("Applied ParentCommand filtering to LangGraph agent span")
        except Exception as e:
            logger.debug(f"Failed to apply ParentCommand filtering: {e}")

    # In multi agent scenarios, the root agent is the one that orchestrates the other agents. LangGraph generates an extra root level invoke()
    # call on top of the supervisor agent invoke().
    # This span handler resets the parent invoke call as generic type to avoid duplicate attributes/events in supervisor span and this root span.

    def hydrate_span(self, to_wrap, wrapped, instance, args, kwargs, result, span, parent_span = None, ex:Exception = None, is_post_exec:bool= False) -> bool:
        # Filter out ParentCommand exceptions as they are LangGraph control flow mechanisms, not actual errors
        if ParentCommand is not None and isinstance(ex, ParentCommand):
            ex = None  # Suppress the ParentCommand exception from being recorded
            
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
    def post_task_processing(self, to_wrap, wrapped, instance, args, kwargs, result, ex, span, parent_span):
        """Apply ParentCommand filtering to the span before task execution."""
        # Apply ParentCommand filtering to this span
        self._apply_parent_command_filtering(span)
        super().post_task_processing(to_wrap, wrapped, instance, args, kwargs, result, ex, span, parent_span)

    def _apply_parent_command_filtering(self, span):
        """Apply ParentCommand exception filtering to a span."""
        try:
            if hasattr(span, 'record_exception'):
                # Create a filtered wrapper and replace the record_exception method
                filter_wrapper = ParentCommandFilterSpan(span)
                span.record_exception = filter_wrapper.record_exception
                logger.debug("Applied ParentCommand filtering to LangGraph tool span")
        except Exception as e:
            logger.debug(f"Failed to apply ParentCommand filtering: {e}")
    
    def hydrate_span(self, to_wrap, wrapped, instance, args, kwargs, result, span, parent_span = None, ex:Exception = None, is_post_exec:bool= False) -> bool:
        # Filter out ParentCommand exceptions as they are LangGraph control flow mechanisms, not actual errors
        if ParentCommand is not None and isinstance(ex, ParentCommand):
            ex = None  # Suppress the ParentCommand exception from being recorded
            
        if is_delegation_tool(instance):
            agent_request_wrapper = to_wrap.copy()
            agent_request_wrapper["output_processor"] = AGENT_DELEGATION
        else:
            agent_request_wrapper = to_wrap

        return super().hydrate_span(agent_request_wrapper, wrapped, instance, args, kwargs, result, span, parent_span, ex, is_post_exec)
    