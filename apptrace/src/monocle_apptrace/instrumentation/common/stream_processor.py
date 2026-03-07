"""
Base streaming processor using Template Method pattern for generic framework support.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

from monocle_apptrace.instrumentation.common.utils import patch_instance_method


@dataclass
class StreamState:
    """State object for tracking streaming response processing."""
    waiting_for_first_token: bool = True
    first_token_time: int = 0
    stream_closed_time: Optional[int] = None
    accumulated_response: str = ""
    token_usage: Optional[Any] = None
    raw_items: List[Any] = field(default_factory=list)
    finish_reason: Optional[str] = None
    role: str = "assistant"
    tools: List[Dict[str, Any]] = field(default_factory=list)
    refusal: Optional[str] = None
    reasoning_content: str = ""
    
    def update_first_token_time(self) -> None:
        """Update first token timestamp if still waiting for first token."""
        if self.waiting_for_first_token:
            self.waiting_for_first_token = False
            self.first_token_time = time.time_ns()
    
    def add_content(self, content: str) -> None:
        """Add content to accumulated response and update first token time."""
        if content:
            self.update_first_token_time()
            self.accumulated_response += content
    
    def store_chunk_or_event(self, item: Any) -> None:
        """Store chunk or event for post-processing."""
        self.raw_items.append(item)
    
    def close_stream(self) -> None:
        """Mark stream as closed with current timestamp."""
        self.stream_closed_time = time.time_ns()


class BaseStreamProcessor(ABC):
    """Base class for streaming processors using Template Method pattern.
    
    This class provides a structured approach to processing streaming responses
    from various AI/ML frameworks. Subclasses should implement the abstract methods
    and optionally override the configurable methods as needed.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    # =============================================================================
    # PUBLIC API - Main entry point (do not override)
    # =============================================================================
    
    def process_stream(self, to_wrap: bool, response: Any, span_processor: Optional[Callable]) -> None:
        """Template method for processing streaming responses."""
        stream_start_time = time.time_ns()
        state = self.initialize_state(stream_start_time)
        
        if to_wrap and hasattr(response, "__iter__"):
            self._wrap_sync_iterator(response, state, stream_start_time, span_processor)
        
        if to_wrap and hasattr(response, "__aiter__"):
            self._wrap_async_iterator(response, state, stream_start_time, span_processor)
    
    # =============================================================================
    # ABSTRACT METHODS - Must be implemented by subclasses
    # =============================================================================
    
    @abstractmethod
    def handle_event(self, item: Any, state: StreamState) -> bool:
        """Handle event-based fragment format (e.g., response.* events).
        
        Args:
            item: Stream item that might be an event
            state: Current streaming state
            
        Returns:
            True if item was recognized and processed, False otherwise
        """
        pass
    
    @abstractmethod
    def handle_chunk(self, item: Any, state: StreamState) -> bool:
        """Handle chunked fragment format with delta objects.
        
        Args:
            item: Stream item that might be a chunk with delta
            state: Current streaming state
            
        Returns:
            True if item was recognized and processed, False otherwise
        """
        pass
    
    @abstractmethod
    def handle_completion(self, item: Any, state: StreamState) -> bool:
        """Handle completion metadata chunks (usage info, finish_reason, etc.).
        
        Args:
            item: Stream item that might contain completion metadata
            state: Current streaming state
            
        Returns:
            True if item was recognized and processed, False otherwise
        """
        pass
    
    # =============================================================================
    # CONFIGURABLE METHODS - May be overridden by subclasses as needed
    # =============================================================================
    
    def initialize_state(self, stream_start_time: int) -> StreamState:
        """Initialize the streaming state.
        
        Override this method to customize the initial state structure
        or add framework-specific state variables.
        
        Args:
            stream_start_time: Timestamp when streaming started
            
        Returns:
            StreamState object containing the initial streaming state
        """
        state = StreamState()
        state.first_token_time = stream_start_time
        return state
    
    def apply_generic_processing(self, item: Any, state: StreamState) -> None:
        """Apply generic fallback processing when framework-specific processing fails.
        
        This is executed as a last resort when none of the abstract methods
        successfully process the item. Override to customize fallback behavior.
        
        Args:
            item: Stream item that wasn't handled by framework-specific methods
            state: Current streaming state
        """
        # Default generic processing - can be extended
        if hasattr(item, 'content') and item.content:
            state.add_content(str(item.content))
    
    def handle_processing_error(self, error: Exception, item: Any, state: StreamState) -> None:
        """Handle processing errors during stream processing.
        
        Override to customize error handling behavior (e.g., different logging,
        error recovery strategies, etc.).
        
        Args:
            error: Exception that occurred during processing
            item: Stream item that caused the error
            state: Current streaming state
        """
        self.logger.warning(
            "Warning: Error occurred while processing stream item: %s", str(error)
        )
    
    def store_chunk_or_event(self, item: Any, state: StreamState) -> None:
        """Store streaming chunk or event for post-processing.
        
        Override to customize how streaming chunks/events are stored for later assembly
        (e.g., filtering, transformation, different storage strategies).
        
        Args:
            item: Streaming chunk or event to store
            state: Current streaming state
        """
        state.store_chunk_or_event(item)
    
    def assemble_data(self, state: StreamState) -> None:
        """Assemble fragmented data from stored raw streaming items.
        
        Called after streaming completes to reconstruct complex objects
        like tool calls that arrive as fragments across multiple chunks.
        Override to implement framework-specific data assembly logic.
        
        Args:
            state: Final streaming state with all raw items stored
        """
        pass
    
    def build_span_result(self, state: StreamState, stream_start_time: int) -> SimpleNamespace:
        """Build the final span result.
        
        Override to customize the structure or content of the final result
        (e.g., additional metadata, different timestamp handling, etc.).
        
        Args:
            state: Final streaming state after assembly
            stream_start_time: Timestamp when streaming started
            
        Returns:
            SimpleNamespace containing the span result
        """
        return SimpleNamespace(
            type="stream",
            timestamps={
                "role": state.role,
                "data.input": int(stream_start_time),
                "data.output": int(state.first_token_time),
                "metadata": int(state.stream_closed_time or time.time_ns()),
            },
            output_text=state.accumulated_response,
            tools=state.tools if state.tools else None,
            usage=state.token_usage,
            finish_reason=state.finish_reason,
            refusal=state.refusal,
            reasoning_content=state.reasoning_content,
        )
    
    # =============================================================================
    # INTERNAL IMPLEMENTATION - Do not override these methods
    # =============================================================================
    
    def _wrap_sync_iterator(self, response: Any, state: StreamState, 
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
    
    def _wrap_async_iterator(self, response: Any, state: StreamState,
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
    
    def process_fragment(self, item: Any, state: StreamState) -> None:
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
            self.store_chunk_or_event(item, state)
    
    def try_framework_specific_processing(self, item: Any, state: StreamState) -> bool:
        """Try to process item using framework-specific logic.
        
        This template method tries each streaming format in order:
        1. Event-based streaming (response.* events)
        2. Chunked streaming (delta objects)
        3. Completion metadata (usage, finish_reason)
        
        Returns:
            True if item was recognized and processed, False otherwise
        """
        # Try event-based streaming first
        if self.handle_event(item, state):
            return True
        
        # Try chunked streaming format
        if self.handle_chunk(item, state):
            return True
        
        # Try completion metadata
        if self.handle_completion(item, state):
            return True
        
        return False
    
    def create_span_result(self, state: StreamState, stream_start_time: int) -> SimpleNamespace:
        """Template method for creating span result."""
        # Assemble fragmented data from accumulated items
        self.assemble_data(state)
        
        # Create result with framework-specific data
        return self.build_span_result(state, stream_start_time)