"""
Base streaming processor using Template Method pattern for generic framework support.
"""

import logging
import time
from abc import ABC, abstractmethod
from types import SimpleNamespace
from typing import Any, Callable, Dict, Optional

from monocle_apptrace.instrumentation.common.utils import patch_instance_method


class BaseStreamProcessor(ABC):
    """Base class for streaming processors using Template Method pattern."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def process_stream(self, to_wrap: bool, response: Any, span_processor: Optional[Callable]) -> None:
        """Template method for processing streaming responses."""
        stream_start_time = time.time_ns()
        state = self.initialize_state(stream_start_time)
        
        if to_wrap and hasattr(response, "__iter__"):
            self._wrap_sync_iterator(response, state, stream_start_time, span_processor)
        
        if to_wrap and hasattr(response, "__aiter__"):
            self._wrap_async_iterator(response, state, stream_start_time, span_processor)
    
    def initialize_state(self, stream_start_time: int) -> Dict[str, Any]:
        """Initialize the streaming state. Can be overridden by subclasses."""
        return {
            "waiting_for_first_token": True,
            "first_token_time": stream_start_time,
            "stream_closed_time": None,
            "accumulated_response": "",
            "token_usage": None,
            "accumulated_temp_list": [],
            "finish_reason": None,
            "role": "assistant",
            "tools": [],
            "refusal": None,
            "reasoning_content": "",
        }
    
    def _wrap_sync_iterator(self, response: Any, state: Dict[str, Any], 
                           stream_start_time: int, span_processor: Optional[Callable]) -> None:
        """Wrap synchronous iterator."""
        original_iter = response.__iter__
        
        def new_iter(self_iter):
            for item in original_iter():
                self.process_fragment(item, state)
                yield item
            
            if span_processor:
                ret_val = self.create_span_result(state, stream_start_time)
                span_processor(ret_val)
        
        patch_instance_method(response, "__iter__", new_iter)
    
    def _wrap_async_iterator(self, response: Any, state: Dict[str, Any],
                            stream_start_time: int, span_processor: Optional[Callable]) -> None:
        """Wrap asynchronous iterator."""
        original_iter = response.__aiter__
        
        async def new_aiter(self_iter):
            async for item in original_iter():
                self.process_fragment(item, state)
                yield item
            
            if span_processor:
                ret_val = self.create_span_result(state, stream_start_time)
                span_processor(ret_val)
        
        patch_instance_method(response, "__aiter__", new_aiter)
    
    def process_fragment(self, item: Any, state: Dict[str, Any]) -> None:
        """Template method for processing a single stream fragment."""
        try:
            # Try framework-specific processing first
            if self.try_framework_specific_processing(item, state):
                return  # Successfully processed
            
            # Fall back to generic processing
            self.apply_generic_processing(item, state)
        except Exception as e:
            self.handle_processing_error(e, item, state)
        finally:
            self.store_item(item, state)
    
    def try_framework_specific_processing(self, item: Any, state: Dict[str, Any]) -> bool:
        """Try to process item using framework-specific logic.
        
        This template method tries each streaming format in order:
        1. Event-based streaming (response.* events)
        2. Chunked streaming (delta objects)
        3. Completion metadata (usage, finish_reason)
        
        Returns:
            True if item was recognized and processed, False otherwise
        """
        # Try event-based streaming first
        if self.handle_event_fragment(item, state):
            return True
        
        # Try chunked streaming format
        if self.handle_chunked_fragment(item, state):
            return True
        
        # Try completion metadata
        if self.handle_completion_metadata(item, state):
            return True
        
        return False
    
    @abstractmethod
    def handle_event_fragment(self, item: Any, state: Dict[str, Any]) -> bool:
        """Handle event-based fragment format (e.g., response.* events).
        
        Returns:
            True if item was recognized and processed, False otherwise
        """
        pass
    
    @abstractmethod
    def handle_chunked_fragment(self, item: Any, state: Dict[str, Any]) -> bool:
        """Handle chunked fragment format with delta objects.
        
        Returns:
            True if item was recognized and processed, False otherwise
        """
        pass
    
    @abstractmethod
    def handle_completion_metadata(self, item: Any, state: Dict[str, Any]) -> bool:
        """Handle completion metadata chunks (usage info, finish_reason, etc.).
        
        Returns:
            True if item was recognized and processed, False otherwise
        """
        pass
    
    def apply_generic_processing(self, item: Any, state: Dict[str, Any]) -> None:
        """Apply generic fallback processing when framework-specific processing fails.
        
        This is always executed as a last resort. Can be overridden.
        """
        # Default generic processing - can be extended
        if hasattr(item, 'content') and item.content:
            if state["waiting_for_first_token"]:
                state["waiting_for_first_token"] = False
                state["first_token_time"] = time.time_ns()
            state["accumulated_response"] += str(item.content)

    
    def handle_processing_error(self, error: Exception, item: Any, state: Dict[str, Any]) -> None:
        """Handle processing errors. Can be overridden."""
        self.logger.warning(
            "Warning: Error occurred while processing stream item: %s", str(error)
        )
    
    def store_item(self, item: Any, state: Dict[str, Any]) -> None:
        """Store item for post-processing. Can be overridden."""
        state["accumulated_temp_list"].append(item)

    
    def create_span_result(self, state: Dict[str, Any], stream_start_time: int) -> SimpleNamespace:
        """Template method for creating span result."""
        # Assemble fragmented data from accumulated items
        self.assemble_fragmented_data(state)
        
        # Create result with framework-specific data
        return self.build_span_result(state, stream_start_time)
    
    def assemble_fragmented_data(self, state: Dict[str, Any]) -> None:
        """Assemble fragmented data from accumulated streaming items.
        
        Called after streaming completes to reconstruct complex objects
        like tool calls that arrive as fragments across multiple chunks.
        Can be overridden by subclasses.
        """
        pass
    
    def build_span_result(self, state: Dict[str, Any], stream_start_time: int) -> SimpleNamespace:
        """Build the final span result. Can be overridden."""
        return SimpleNamespace(
            type="stream",
            timestamps={
                "role": state["role"],
                "data.input": int(stream_start_time),
                "data.output": int(state["first_token_time"]),
                "metadata": int(state["stream_closed_time"] or time.time_ns()),
            },
            output_text=state["accumulated_response"],
            tools=state["tools"] if state["tools"] else None,
            usage=state["token_usage"],
            finish_reason=state["finish_reason"],
            refusal=state.get("refusal"),
            reasoning_content=state.get("reasoning_content"),
        )