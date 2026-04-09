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
        self._post_process_span = post_process_span
        self._finalized = False
    
    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to wrapped result, except for stream_events."""
        if name in ('_streaming_result', '_post_process_span', '_finalized'):
            return object.__getattribute__(self, name)
        return getattr(self._streaming_result, name)
    
    async def stream_events(self):
        """Wrap stream_events to capture final_output after iteration completes."""
        try:
            # Iterate through all streamed events from the wrapped result
            async for event in self._streaming_result.stream_events():
                yield event
            
            # After iteration completes, finalize the span with output
            if not self._finalized:
                self._finalized = True
                self._finalize_span()
        except Exception as e:
            logger.debug(f"Error in wrapped stream_events: {e}")
            raise
    
    def _finalize_span(self):
        """Capture final output and finalize the span."""
        try:
            # Get the final output that should now be available
            final_output = getattr(self._streaming_result, 'final_output', None)
            if final_output is not None:
                logger.debug(f"Captured final_output from streaming: {final_output}")
                # Call post_process_span with self (the wrapper) - 
                # extract_agent_response will be able to get final_output
                self._post_process_span(self)
            else:
                logger.debug("No final_output available after streaming")
                self._post_process_span(self)
        except Exception as e:
            logger.debug(f"Error finalizing span: {e}")
            try:
                self._post_process_span(self)
            except Exception as e2:
                logger.debug(f"Error calling post_process_span: {e2}")
    
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
        # No response to wrap
        return None
    
    # Check if this is a RunResultStreaming object
    if type(response).__name__ != "RunResultStreaming":
        # Not a streaming result, don't wrap
        return None
    
    try:
        # Wrap the streaming result
        wrapper = StreamingResultWrapper(response, span_processor)
        return wrapper
    except Exception as e:
        logger.warning(f"Error wrapping streaming response: {e}")
        # Fall back to immediate processing
        span_processor(response)
        return None
