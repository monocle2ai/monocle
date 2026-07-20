import logging
from types import SimpleNamespace
from opentelemetry.context import attach, set_value, detach
from opentelemetry.trace import Tracer
from monocle_apptrace.instrumentation.common.constants import AGENT_NAME_KEY, AGENT_PREFIX_KEY, SPAN_TYPES
from monocle_apptrace.instrumentation.common.span_handler import (
    SpanHandler as BaseSpanHandler,
)
from monocle_apptrace.instrumentation.common.utils import with_tracer_wrapper, propogate_agent_name_to_parent_span, get_scopes, get_current_monocle_span

# Scope an app opens to collapse its many Runner.run calls into one turn (see skip_span).
AGENT_TURN_SCOPE = "agentic.turn"
from monocle_apptrace.instrumentation.common.wrapper import atask_wrapper
from monocle_apptrace.instrumentation.metamodel.agents._helper import (
    AGENTS_AGENT_NAME_KEY,
    DELEGATION_NAME_PREFIX,
    get_runner_agent_name,
    get_agent_name,
    extract_agent_response,
)
from monocle_apptrace.instrumentation.metamodel.agents.entities.inference import (
    AGENT_DELEGATION,
)

logger = logging.getLogger(__name__)


@with_tracer_wrapper
def constructor_wrapper(
    tracer: Tracer,
    handler: BaseSpanHandler,
    to_wrap,
    wrapped,
    instance,
    source_path,
    args,
    kwargs,
):

    original_func = kwargs.get("on_invoke_tool", None)
    result = None
    mcp_url = None
    # kwargs.get("on_invoke_tool").args[0].params["url"]
    if (
        kwargs.get("on_invoke_tool")
        and hasattr(kwargs.get("on_invoke_tool"), "args")
        and len(kwargs.get("on_invoke_tool").args) > 0
        and hasattr(kwargs.get("on_invoke_tool").args[0], "params")
    ):
        mcp_url = kwargs.get("on_invoke_tool").args[0].params.get("url", None)
        if mcp_url:
            logger.debug(f"Using MCP URL: {mcp_url}")
    tool_instance = SimpleNamespace(
        name=kwargs.get("name", "unknown_tool"),
        description=kwargs.get("description", "No description provided"),
    )
    if original_func and not getattr(original_func, "_monocle_wrapped", False):
        # Now wrap the function with our instrumentation

        async def wrapped_func(*func_args, **func_kwargs):
            token = None
            try:
                if mcp_url:
                    token = attach(set_value("mcp.url", mcp_url))
                # Use the handler to create spans when the decorated function is called
                return await atask_wrapper(
                    tracer=tracer, handler=handler, to_wrap=to_wrap
                )(
                    wrapped=original_func,
                    instance=tool_instance,
                    source_path=source_path,
                    args=func_args,
                    kwargs=func_kwargs,
                )
            finally:
                if token:
                    detach(token)

        kwargs["on_invoke_tool"] = wrapped_func
        # Preserve function metadata
        wrapped_func.__name__ = getattr(wrapped, "__name__", "unknown_tool")
        wrapped_func.__doc__ = getattr(wrapped, "__doc__", "")
        # mark function as wrapped
        setattr(wrapped_func, "_monocle_wrapped", True)

    result = wrapped(*args, **kwargs)
    return result


@with_tracer_wrapper
def handoff_constructor_wrapper(
    tracer: Tracer,
    handler: BaseSpanHandler,
    to_wrap,
    wrapped,
    instance,
    source_path,
    args,
    kwargs,
):

    original_func = kwargs.get("on_invoke_handoff", None)
    result = None
    tool_instance = SimpleNamespace(
        name=kwargs.get("name", "unknown_handoff"),
        description=kwargs.get("description", "No description provided"),
    )
    if original_func and not getattr(original_func, "_monocle_wrapped", False):
        # Now wrap the function with our instrumentation
        async def wrapped_func(*func_args, **func_kwargs):
            # Use the handler to create spans when the decorated function is called
            return await atask_wrapper(tracer=tracer, handler=handler, to_wrap=to_wrap)(
                wrapped=original_func,
                instance=tool_instance,
                source_path=source_path,
                args=func_args,
                kwargs=func_kwargs,
            )

        kwargs["on_invoke_handoff"] = wrapped_func
        # Preserve function metadata
        wrapped_func.__name__ = getattr(wrapped, "__name__", "unknown_handoff")
        wrapped_func.__doc__ = getattr(wrapped, "__doc__", "")
        # mark function as wrapped
        setattr(wrapped_func, "_monocle_wrapped", True)

    result = wrapped(*args, **kwargs)
    return result


class AgentsSpanHandler(BaseSpanHandler):
    """Span handler for OpenAI Agents SDK."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_context_token = None

    def skip_span(self, to_wrap, wrapped, instance, args, kwargs) -> bool:
        """Skip spans that would otherwise be duplicated.

        1. Streaming dedup: run_streamed already emits the turn + invocation spans via
           its output_processor_list, and internally calls run_single_turn_streamed.
           Skip that inner call so the invocation isn't traced twice.
        2. Turn-collapse: skip the Runner.run turn span when an agentic.turn scope is
           already open, so an app's many Runner.run calls nest as invocations under one
           turn. Only the turn entry is skipped (run_single_turn invocations are not);
           inert without the scope.
        """
        method_name = to_wrap.get("method", "")
        if method_name == "run_single_turn_streamed":
            parent_span = get_current_monocle_span()
            parent_name = getattr(parent_span, "name", "") if parent_span else ""
            if parent_name.endswith("run_streamed"):
                return True

        output_processor = to_wrap.get("output_processor") or {}
        is_turn = output_processor.get("type") == SPAN_TYPES.AGENTIC_REQUEST
        if is_turn and AGENT_TURN_SCOPE in get_scopes():
            return True
        return super().skip_span(to_wrap, wrapped, instance, args, kwargs)

    def _get_agent_name(self, to_wrap, wrapped, instance, args, kwargs):
        """Set the agent context for tracking across calls."""
        try:
            # For Runner.run, the agent is the first argument
            if len(args) > 0:
                agent = args[0]
                agent_name = get_runner_agent_name(agent)
                return agent_name
        except Exception as e:
            logger.warning("Warning: Error setting agent context: %s", str(e))
        return ""

    def pre_tracing(self, to_wrap, wrapped, instance, args, kwargs):
        """Pre-tracing for agent tasks."""
        agent_name = get_agent_name(args, kwargs)
        context = set_value(AGENT_NAME_KEY, agent_name)
        context = set_value(AGENT_PREFIX_KEY, DELEGATION_NAME_PREFIX, context)
        return attach(context), None

    def post_task_processing(self, to_wrap, wrapped, instance, args, kwargs, result, ex, span, parent_span):
        """Post-processing for agent tasks."""
        propogate_agent_name_to_parent_span(span, parent_span)

        try:
            if (
                span is not None
                and parent_span is not None
                and span.attributes.get("span.type") == "agentic.invocation"
                and parent_span.attributes.get("span.type") == "agentic.turn"
            ):
                response = extract_agent_response(result)
                if response:
                    updated = False
                    for event in getattr(parent_span, "events", []):
                        if event.name == "data.output":
                            if hasattr(event, "attributes") and hasattr(event.attributes, "_dict"):
                                event.attributes._dict["response"] = response
                                updated = True
                                break

                    if not updated and hasattr(parent_span, "add_event"):
                        parent_span.add_event("data.output", attributes={"response": response})
        except Exception as e:
            logger.debug(f"Could not copy invocation response to turn span: {e}")

        super().post_task_processing(to_wrap, wrapped, instance, args, kwargs, result, ex, span, parent_span)