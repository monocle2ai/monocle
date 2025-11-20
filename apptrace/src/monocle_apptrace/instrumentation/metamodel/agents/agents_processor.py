import logging
from types import SimpleNamespace
from opentelemetry.context import attach, set_value, detach
from opentelemetry.trace import Tracer
from monocle_apptrace.instrumentation.common.constants import AGENT_NAME_KEY, AGENT_PREFIX_KEY
from monocle_apptrace.instrumentation.common.span_handler import (
    SpanHandler as BaseSpanHandler,
)
from monocle_apptrace.instrumentation.common.utils import with_tracer_wrapper, propogate_agent_name_to_parent_span
from monocle_apptrace.instrumentation.common.wrapper import atask_wrapper
from monocle_apptrace.instrumentation.metamodel.agents._helper import (
    AGENTS_AGENT_NAME_KEY,
    DELEGATION_NAME_PREFIX,
    get_runner_agent_name,
    get_agent_name
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
        return attach(context)

    def post_task_processing(self, to_wrap, wrapped, instance, args, kwargs, result, ex, span, parent_span):
        """Post-processing for agent tasks."""
        propogate_agent_name_to_parent_span(span, parent_span)
        super().post_task_processing(to_wrap, wrapped, instance, args, kwargs, result, ex, span, parent_span)
