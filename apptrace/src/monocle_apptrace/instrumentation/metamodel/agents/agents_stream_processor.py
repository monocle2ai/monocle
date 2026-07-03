"""Stream processor for OpenAI Agents SDK streaming responses."""

import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class StreamingResultWrapper:
    """Wraps RunResultStreaming to capture final_output after stream iteration.

    This wrapper stays transparent to the user - all attribute access is delegated
    to the wrapped RunResultStreaming object. The only difference is that
    stream_events() is hooked to capture the final_output and finalize the span
    when streaming completes.
    """

    def __init__(self, streaming_result: Any, post_process_span: Callable):
        """Initialize wrapper.

        Args:
            streaming_result: The RunResultStreaming object from run_streamed()
            post_process_span: Function to call when streaming completes
        """
        self._streaming_result = streaming_result
        # A single streamed run can back more than one span (e.g. the agent turn
        # span wrapping the streamed invocation span). Each nested span registers
        # its own finalizer here; they run in registration order (innermost first)
        # once the stream is consumed, so a parent span is still open when a child
        # finalizes into it.
        self._finalizers = [post_process_span]
        self._finalized = False

    def add_finalizer(self, post_process_span: Callable) -> None:
        """Register an additional span finalizer to run when streaming completes."""
        self._finalizers.append(post_process_span)

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to wrapped result, except for stream_events."""
        if name in ('_streaming_result', '_finalizers', '_finalized'):
            return object.__getattribute__(self, name)
        return getattr(self._streaming_result, name)

    async def stream_events(self):
        """Wrap stream_events to capture final_output after iteration completes."""
        try:
            async for event in self._streaming_result.stream_events():
                yield event

            if not self._finalized:
                self._finalized = True
                self._finalize_span()
        except Exception as e:
            logger.debug(f"Error in wrapped stream_events: {e}")
            raise

    def _finalize_span(self):
        """Capture final output and finalize every registered span."""
        try:
            final_output = getattr(self._streaming_result, 'final_output', None)
            if final_output is not None:
                logger.debug(f"Captured final_output from streaming: {final_output}")
            else:
                logger.debug("No final_output available after streaming")
        except Exception as e:
            logger.debug(f"Error reading final_output from streaming: {e}")
        # Run each registered finalizer independently so one failing span doesn't
        # prevent the others from closing (innermost invocation first, turn last).
        for finalize in self._finalizers:
            try:
                finalize(self)
            except Exception as e:
                logger.debug(f"Error finalizing span: {e}")

    def __repr__(self):
        return f"StreamingResultWrapper({self._streaming_result!r})"


def process_agent_stream(to_wrap: Any, response: Any, span_processor: Callable) -> Optional[Any]:
    """Response processor for OpenAI Agents SDK streaming.

    This is called by the instrumentation framework when run_streamed returns.
    It wraps the RunResultStreaming object to capture final_output when the
    user consumes the stream via stream_events().

    Args:
        to_wrap: The instrumentation configuration dict
        response: The RunResultStreaming object from run_streamed()
        span_processor: Function to call when span should be finalized

    Returns:
        A StreamingResultWrapper that the user will interact with
    """
    if response is None:
        return None

    # An outer span (e.g. the agent turn) sees the already-wrapped result returned
    # by its inner span; chain its finalizer onto the same wrapper instead of
    # re-wrapping, so both spans close once the stream is consumed.
    if isinstance(response, StreamingResultWrapper):
        response.add_finalizer(span_processor)
        return response

    if type(response).__name__ != "RunResultStreaming":
        return None

    try:
        wrapper = StreamingResultWrapper(response, span_processor)
        return wrapper
    except Exception as e:
        logger.warning(f"Error wrapping streaming response: {e}")
        span_processor(response)
        return None
