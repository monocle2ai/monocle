"""
Base test class and fixtures for agent testing in monocle.

This module provides the BaseAgentTest class and related pytest fixtures
for simplified agent testing with trace validation.
"""
import logging
from typing import Any, Callable

import pytest

from monocle_tfwk.assertions import TraceAssertions
from monocle_tfwk.validator import MonocleValidator
from monocle_tfwk.visualization.gantt_chart import TraceGanttChart, VisualizationMode

logger = logging.getLogger(__name__)


class BaseAgentTest:
    """Base class for agent tests, similar to AgentiTest's BaseAgentTest."""

    def display_flow_gantt_chart(self, mode: VisualizationMode = VisualizationMode.DETAILED) -> None:
        """
        Display Gantt chart visualization of the execution flow with configurable options.
        
        Args:
            mode: Visualization mode from VisualizationMode enum
            config: Custom configuration dict (currently unused - for future extensibility)
            
        Modes:
            - VisualizationMode.FULL: Complete report with timeline, critical path, mermaid, and attributes
            - VisualizationMode.COMPACT: Timeline only without attributes for quick overview
            - VisualizationMode.DETAILED: Attributes-focused view for deep analysis
            
        Examples:
            self.display_flow_gantt_chart()  # Default detailed report
            self.display_flow_gantt_chart(VisualizationMode.COMPACT)  # Quick overview
            self.display_flow_gantt_chart(VisualizationMode.FULL)  # Full report with everything
        """
        try:
            # Get spans from traces - traces is a TraceAssertions object
            spans = self.assert_traces().spans  # Ensure we have the latest spans
            
            if not spans:
                logger.info("No spans available for visualization")
                return
            
            # Create Gantt chart
            gantt = TraceGanttChart(spans)
            
            # Parse spans and handle any timing issues
            try:
                events = gantt.parse_spans()
                logger.info(f"📈 Successfully parsed {len(events)} timeline events")
                
                # Generate visualization based on mode
                output = gantt.generate_visualization_report(mode)
                
                # Log with appropriate message based on mode
                if mode == VisualizationMode.COMPACT:
                    logger.info("📈 Compact Trace View:")
                elif mode == VisualizationMode.DETAILED:
                    logger.info("📈 Detailed Trace Analysis:")
                else:  # VisualizationMode.FULL
                    logger.info("📈 Comprehensive Visualization Report:")
                
                logger.info("\n" + output)
                
            except Exception as parse_error:
                logger.warning(f"⚠️ Could not parse spans for Gantt chart: {parse_error}")
                # Try to show basic span information instead
                logger.info("📋 Basic Span Information:")
                for span in spans:
                    span_name = getattr(span, 'name', 'Unknown')
                    span_type = getattr(span, 'attributes', {}).get('span.type', 'Unknown')
                    logger.info(f"  • {span_name} ({span_type})")
                
        except ImportError as e:
            logger.warning(f"⚠️ Visualization modules not available: {e}")
        except Exception as e:
            logger.warning(f"⚠️ Could not generate Gantt chart: {e}")
            import traceback
            logger.debug(f"Full traceback: {traceback.format_exc()}")
    
    def display_compact_flow(self) -> None:
        """Display a compact trace view without attributes for quick overview."""
        self.display_flow_gantt_chart(mode=VisualizationMode.COMPACT)
    
    def display_detailed_flow(self) -> None:
        """Display a detailed trace view with all attributes for analysis."""
        self.display_flow_gantt_chart(mode=VisualizationMode.DETAILED)
    
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
            
    def assert_traces(self) -> TraceAssertions:
        """Get trace assertions for the current test's spans."""
        spans = self.validator.spans
        return TraceAssertions(spans)
        
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
        return self.assert_traces().assert_agent(agent_name)
        
    def assert_tool_called(self, tool_name: str):
        """Simple assertion that a tool was called.""" 
        return self.assert_traces().called_tool(tool_name)
        
    def assert_no_errors(self):
        """Assert no errors occurred in any spans."""
        return self.assert_traces().completed_successfully()
        
    def assert_performance(self, max_duration: float):
        """Assert all operations completed within time limit."""
        return self.assert_traces().within_time_limit(max_duration)


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


