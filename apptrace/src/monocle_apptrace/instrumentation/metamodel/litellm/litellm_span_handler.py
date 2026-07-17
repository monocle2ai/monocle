"""
LiteLLM Async Span Handler for setting async context flags.
"""

from monocle_apptrace.instrumentation.common.span_handler import SpanHandler
from opentelemetry.context import attach, get_current, get_value, set_value, Context

class LiteLLMSyncSpanHandler(SpanHandler):
    """
    Span handler for LiteLLM sync operations.
    Skips span creation when LITELLM_ASYNC flag is set to true.
    """

    def skip_span(self, to_wrap, wrapped, instance, args, kwargs) -> bool:
        """
        Skip span creation if LITELLM_ASYNC is set to true in the context.
        
        Args:
            to_wrap: The wrapper configuration
            wrapped: The wrapped function
            instance: The instance being wrapped
            args: Positional arguments
            kwargs: Keyword arguments
            
        Returns:
            bool: True if the argument acompletion is true, False otherwise
        """
        # response_api_handler signals its async variant via _is_async: the
        # sync call then returns an unawaited coroutine, so a span here would
        # never see the real result (the async wrap owns the span).
        return kwargs.get('acompletion') is True or kwargs.get('_is_async') is True

