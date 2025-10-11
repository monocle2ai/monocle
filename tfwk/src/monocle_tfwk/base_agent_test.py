"""
Base test class and fixtures for agent testing in monocle.

This module provides the BaseAgentTest class and related pytest fixtures
for simplified agent testing with trace validation.
"""
import logging
from typing import Any, Callable

import pytest

from .assertions import TraceAssertions
from .validator import MonocleValidator

logger = logging.getLogger(__name__)


class BaseAgentTest:
    """Base class for agent tests, similar to AgentiTest's BaseAgentTest."""
    
    @pytest.fixture(autouse=True)
    def setup_test(self, request):
        """Auto-used fixture to set up test context."""
        self.validator = MonocleValidator()
        self.test_name = request.node.name
        # Clear any previous spans
        self.validator.clear_spans()
        
        # Add teardown to properly clean up contexts
        def teardown():
            try:
                # Clear spans again on teardown
                self.validator.clear_spans()
            except Exception as e:
                logger.debug(f"Cleanup warning (non-critical): {e}")
        
        request.addfinalizer(teardown)
            
    def assert_trace(self) -> TraceAssertions:
        """Get trace assertions for the current test's spans."""
        spans = self.validator.spans
        return TraceAssertions(spans)
        
    def assert_traces(self) -> TraceAssertions:
        """Get trace assertions for the current test's spans. Alias for assert_trace()."""
        return self.assert_trace()
        
    async def run_agent(self, agent_func: Callable, *args, **kwargs) -> Any:
        """Run an agent function and capture its traces."""
        try:
            # Run the agent
            if callable(agent_func):
                if hasattr(agent_func, '__code__') and agent_func.__code__.co_flags & 0x80:  # Check if coroutine
                    result = await agent_func(*args, **kwargs)
                else:
                    result = agent_func(*args, **kwargs)
            else:
                raise ValueError("agent_func must be callable")
                
            return result
        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            raise
            
    def assert_agent_called(self, agent_name: str):
        """Simple assertion that an agent was called."""
        return self.assert_trace().assert_agent(agent_name)
        
    def assert_tool_called(self, tool_name: str):
        """Simple assertion that a tool was called.""" 
        return self.assert_trace().called_tool(tool_name)
        
    def assert_no_errors(self):
        """Assert no errors occurred in any spans."""
        return self.assert_trace().completed_successfully()
        
    def assert_performance(self, max_duration: float):
        """Assert all operations completed within time limit."""
        return self.assert_trace().within_time_limit(max_duration)


# Pytest fixtures for agent testing
@pytest.fixture(scope="function")
def trace_validator():
    """Fixture providing a MonocleValidator instance."""
    return MonocleValidator()


@pytest.fixture(scope="function") 
def agent_test_context(trace_validator):
    """Fixture providing agent test context."""
    return {
        "validator": trace_validator,
        "spans": trace_validator.spans
    }


