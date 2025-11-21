import logging
from opentelemetry.context import set_value, attach, detach, get_value
from monocle_apptrace.instrumentation.common.constants import AGENT_PREFIX_KEY, SCOPE_NAME
from monocle_apptrace.instrumentation.common.span_handler import SpanHandler
from monocle_apptrace.instrumentation.metamodel.crew_ai._helper import (
   DELEGATION_NAME_PREFIX, get_name, is_root_crew_name, is_delegation_task, CREW_AI_AGENT_NAME_KEY
)
from monocle_apptrace.instrumentation.metamodel.crew_ai.entities.inference import (
    AGENT_DELEGATION, AGENT_REQUEST, AGENT
)
from monocle_apptrace.instrumentation.common.scope_wrapper import start_scope, stop_scope
from monocle_apptrace.instrumentation.common.utils import is_scope_set

logger = logging.getLogger(__name__)

class CrewAIAgentHandler(SpanHandler):
    def pre_tracing(self, to_wrap, wrapped, instance, args, kwargs):
        context = set_value(CREW_AI_AGENT_NAME_KEY, get_name(instance))
        context = set_value(AGENT_PREFIX_KEY, DELEGATION_NAME_PREFIX, context)
        scope_name = AGENT_REQUEST.get("type")
        if not is_scope_set(scope_name):
            agent_request_wrapper = to_wrap.copy()
            agent_request_wrapper["output_processor"] = AGENT_REQUEST
            return attach(context), agent_request_wrapper
        else:
            return attach(context), None

    def post_tracing(self, to_wrap, wrapped, instance, args, kwargs, result, token):
        if token is not None:
            detach(token)

    # In multi agent scenarios, the root crew is the one that orchestrates the other agents. CrewAI generates an extra root level kickoff()
    # call on top of the supervisor agent kickoff().
    # This span handler resets the parent kickoff call as generic type to avoid duplicate attributes/events in supervisor span and this root span.

    def hydrate_span(self, to_wrap, wrapped, instance, args, kwargs, result, span, parent_span = None, ex:Exception = None, is_post_exec:bool= False) -> bool:
        return super().hydrate_span(to_wrap, wrapped, instance, args, kwargs, result, span, parent_span, ex, is_post_exec)

class CrewAITaskHandler(SpanHandler):
    def pre_tracing(self, to_wrap, wrapped, instance, args, kwargs):
        if is_delegation_task(instance):
            agent_request_wrapper = to_wrap.copy()
            agent_request_wrapper["output_processor"] = AGENT_DELEGATION
        else:
            agent_request_wrapper = None
        return None, agent_request_wrapper

    # CrewAI uses tasks to coordinate agent execution. The method is task execute() with different task types.
    # Hence we use a different output processor for task execute() to format the span as agentic.delegation when appropriate.
    def post_task_processing(self, to_wrap, wrapped, instance, args, kwargs, result, ex, span, parent_span):
        super().post_task_processing(to_wrap, wrapped, instance, args, kwargs, result, ex, span, parent_span)

    def hydrate_span(self, to_wrap, wrapped, instance, args, kwargs, result, span, parent_span = None, ex:Exception = None, is_post_exec:bool= False) -> bool:
        return super().hydrate_span(to_wrap, wrapped, instance, args, kwargs, result, span, parent_span, ex, is_post_exec)

class CrewAIToolHandler(SpanHandler):
    def pre_tracing(self, to_wrap, wrapped, instance, args, kwargs):
        print(f"🔧 CrewAI Tool Handler: pre_tracing called for {instance.__class__.__name__}")
        print(f"    Method: {to_wrap.get('method')}, Package: {to_wrap.get('package')}")
        print(f"    Args: {args}, Kwargs: {kwargs}")
        return None, None

    def post_tracing(self, to_wrap, wrapped, instance, args, kwargs, result, token):
        print(f"🔧 CrewAI Tool Handler: post_tracing called for {instance.__class__.__name__}")
        print(f"    Result: {result}")
        if token is not None:
            detach(token)

    def hydrate_span(self, to_wrap, wrapped, instance, args, kwargs, result, span, parent_span = None, ex:Exception = None, is_post_exec:bool= False) -> bool:
        print(f"🔧 CrewAI Tool Handler: hydrate_span called for {instance.__class__.__name__}")
        print(f"    Span: {span.name if span else 'None'}, Is_post_exec: {is_post_exec}")
        return super().hydrate_span(to_wrap, wrapped, instance, args, kwargs, result, span, parent_span, ex, is_post_exec)