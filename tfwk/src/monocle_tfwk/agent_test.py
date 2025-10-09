"""
Pytest-style agent testing framework for monocle.

This module provides an intuitive way to test AI agents using pytest fixtures
and base test classes, inspired by AgentiTest but designed for trace-based validation.
"""
import logging
from datetime import datetime
from typing import Any, Callable, Dict, List

import pytest
from opentelemetry.sdk.trace import ReadableSpan

from . import trace_utils
from .semantic_similarity import semantic_similarity
from .validator import MonocleValidator

logger = logging.getLogger(__name__)


def _format_timestamp(nanoseconds: int) -> str:
    """Convert nanosecond timestamp to ISO 8601 format for readable output."""
    seconds = nanoseconds / 1_000_000_000
    dt = datetime.fromtimestamp(seconds)
    return dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')


class TraceAssertions:
    """Fluent assertion API for agent traces with JMESPath query capabilities."""
    
    def __init__(self, spans: List[ReadableSpan]):
        self._all_spans = spans
        self._current_spans = spans
        self._query_engine = None
        
    @property
    def spans(self) -> List[ReadableSpan]:
        """Get the current filtered spans."""
        return self._current_spans
        
    @property
    def all_spans(self) -> List[ReadableSpan]:
        """Get all original spans."""
        return self._all_spans
    
    @property
    def query_engine(self) -> 'trace_utils.TraceQueryEngine':
        """Get the JMESPath query engine for advanced trace analysis."""
        if self._query_engine is None:
            # Create a mock traces object with the spans
            class MockTraces:
                def __init__(self, spans):
                    self.spans = spans
            
            self._query_engine = trace_utils.TraceQueryEngine(MockTraces(self._all_spans))
        return self._query_engine
        
    def assert_agent(self, agent_name: str) -> 'TraceAssertions':
        """Assert that an agent with the given name was invoked."""
        matching_spans = [
            span for span in self._current_spans
            if span.attributes.get("agent.name") == agent_name
        ]
        assert matching_spans, f"No agent named '{agent_name}' found in traces"
        self._current_spans = matching_spans
        return self
        
    def called_tool(self, tool_name: str) -> 'TraceAssertions':
        """Assert that a tool with the given name was called."""
        matching_spans = [
            span for span in self._current_spans
            if span.attributes.get("tool.name") == tool_name
        ]
        assert matching_spans, f"No tool named '{tool_name}' found in traces"
        self._current_spans = matching_spans
        return self
        
    def with_input_containing(self, text: str) -> 'TraceAssertions':
        """Assert that input contains the specified text."""
        matching_spans = []
        for span in self._current_spans:
            input_text = trace_utils.get_input_from_span(span)
            if input_text and text.lower() in input_text.lower():
                matching_spans.append(span)
        assert matching_spans, f"No spans found with input containing '{text}'"
        self._current_spans = matching_spans
        return self
        
    def with_output_containing(self, text: str) -> 'TraceAssertions':
        """Assert that output contains the specified text."""
        matching_spans = []
        for span in self._current_spans:
            output_text = trace_utils.get_output_from_span(span)
            if output_text and text.lower() in output_text.lower():
                matching_spans.append(span)
        assert matching_spans, f"No spans found with output containing '{text}'"
        self._current_spans = matching_spans
        return self
        
    def completed_successfully(self) -> 'TraceAssertions':
        """Assert that spans completed without errors."""
        error_spans = [
            span for span in self._current_spans
            if span.status.status_code.name == "ERROR"
        ]
        assert not error_spans, f"Found {len(error_spans)} spans with errors"
        return self
        
    def within_time_limit(self, max_seconds: float) -> 'TraceAssertions':
        """Assert that operations completed within time limit."""
        slow_spans = []
        for span in self._current_spans:
            duration = (span.end_time - span.start_time) / 1_000_000_000  # Convert to seconds
            if duration > max_seconds:
                slow_spans.append((span, duration))
        assert not slow_spans, f"Found spans exceeding {max_seconds}s: {[(s.name, d) for s, d in slow_spans]}"
        return self
        
    def count(self) -> int:
        """Return the count of current matching spans."""
        return len(self._current_spans)
        
    def exactly(self, count: int) -> 'TraceAssertions':
        """Assert exact count of matching spans."""
        actual_count = len(self._current_spans)
        assert actual_count == count, f"Expected {count} spans, found {actual_count}"
        return self
        
    def at_least(self, count: int) -> 'TraceAssertions':
        """Assert minimum count of matching spans."""
        actual_count = len(self._current_spans)
        assert actual_count >= count, f"Expected at least {count} spans, found {actual_count}"
        return self
        
    def filter_by_name(self, name: str) -> 'TraceAssertions':
        """Filter spans by name."""
        matching_spans = [
            span for span in self._current_spans
            if span.name == name
        ]
        self._current_spans = matching_spans
        return self
        
    def assert_spans(self, min_count: int = None, max_count: int = None, count: int = None) -> 'TraceAssertions':
        """Assert span count conditions."""
        actual_count = len(self._current_spans)
        
        if count is not None:
            assert actual_count == count, f"Expected exactly {count} spans, found {actual_count}"
        elif min_count is not None:
            assert actual_count >= min_count, f"Expected at least {min_count} spans, found {actual_count}"
        elif max_count is not None:
            assert actual_count <= max_count, f"Expected at most {max_count} spans, found {actual_count}"
            
        return self
        
    def assert_span_with_name(self, name: str) -> 'TraceAssertions':
        """Assert that at least one span exists with the given name."""
        matching_spans = [span for span in self._current_spans if span.name == name]
        assert matching_spans, f"No span found with name '{name}'"
        return self
        
    def assert_llm_calls(self, count: int = None, min_count: int = None) -> 'TraceAssertions':
        """Assert LLM call count conditions."""
        llm_spans = [
            span for span in self._current_spans 
            if span.attributes.get("span.type") == "llm" or 
               "llm" in span.name.lower() or
               span.attributes.get("llm.model") is not None
        ]
        
        actual_count = len(llm_spans)
        
        if count is not None:
            assert actual_count == count, f"Expected exactly {count} LLM calls, found {actual_count}"
        elif min_count is not None:
            assert actual_count >= min_count, f"Expected at least {min_count} LLM calls, found {actual_count}"
            
        return self
        
    def assert_attribute(self, key: str, value: str = None) -> 'TraceAssertions':
        """Assert that spans have the specified attribute, optionally with a specific value."""
        matching_spans = []
        for span in self._current_spans:
            if span.attributes and key in span.attributes:
                if value is None or span.attributes.get(key) == value:
                    matching_spans.append(span)
        
        if value is not None:
            assert matching_spans, f"No spans found with attribute '{key}' = '{value}'"
        else:
            assert matching_spans, f"No spans found with attribute '{key}'"
            
        self._current_spans = matching_spans
        return self
        
    def contains_input(self, text: str) -> 'TraceAssertions':
        """Assert that spans contain the specified text in input."""
        return self.with_input_containing(text)
        
    def contains_output(self, text: str) -> 'TraceAssertions':
        """Assert that spans contain the specified text in output."""
        return self.with_output_containing(text)
        
    def semantically_contains_input(self, expected_text: str, threshold: float = 0.75) -> 'TraceAssertions':
        """Assert that spans contain semantically similar text in input using sentence transformers."""
        matching_spans = []
        for span in self._current_spans:
            input_text = trace_utils.get_input_from_span(span)
            if input_text and semantic_similarity(input_text, expected_text, threshold):
                matching_spans.append(span)
        assert matching_spans, f"No spans found with input semantically similar to '{expected_text}' (threshold: {threshold})"
        self._current_spans = matching_spans
        return self
        
    def semantically_contains_output(self, expected_text: str, threshold: float = 0.75) -> 'TraceAssertions':
        """Assert that spans contain semantically similar text in output using sentence transformers."""
        matching_spans = []
        for span in self._current_spans:
            output_text = trace_utils.get_output_from_span(span)
            if output_text and semantic_similarity(output_text, expected_text, threshold):
                matching_spans.append(span)
        assert matching_spans, f"No spans found with output semantically similar to '{expected_text}' (threshold: {threshold})"
        self._current_spans = matching_spans
        return self
        
    def output_semantically_matches(self, expected_text: str, threshold: float = 0.75) -> 'TraceAssertions':
        """Alias for semantically_contains_output for better readability."""
        return self.semantically_contains_output(expected_text, threshold)
        
    def llm_calls(self) -> 'TraceAssertions':
        """Filter to only LLM call spans."""
        llm_spans = [
            span for span in self._current_spans 
            if span.attributes.get("span.type") == "llm" or 
               "llm" in span.name.lower() or
               span.attributes.get("llm.model") is not None
        ]
        self._current_spans = llm_spans
        return self
        
    def filter_by_attribute(self, key: str, value: str = None) -> 'TraceAssertions':
        """Filter spans by attribute key and optionally value."""
        matching_spans = []
        for span in self._current_spans:
            if span.attributes and key in span.attributes:
                if value is None or span.attributes.get(key) == value:
                    matching_spans.append(span)
        self._current_spans = matching_spans
        return self
        
    def debug_spans(self) -> 'TraceAssertions':
        """Print debug information about current spans for troubleshooting."""
        print(f"\n=== DEBUG: {len(self._current_spans)} spans ===")
        for i, span in enumerate(self._current_spans):
            attrs = dict(span.attributes) if span.attributes else {}
            print(f"  [{i}] {span.name}")
            print(f"      Attributes: {attrs}")
            print(f"      Start: {_format_timestamp(span.start_time)}, End: {_format_timestamp(span.end_time)}")
        print("========================\n")
        return self
    
    # JMESPath Query Methods - Direct integration for fluent API
    def query(self, jmes_expression: str) -> Any:
        """Execute a JMESPath query on the trace data.
        
        Args:
            jmes_expression: JMESPath query expression
            
        Returns:
            Query result
        """
        return self.query_engine.query(jmes_expression)
    
    def assert_agent_type(self, agent_type: str) -> 'TraceAssertions':
        """Assert that traces contain a specific agent type using JMESPath."""
        found = self.query_engine.has_agent_type(agent_type)
        assert found, f"Agent type '{agent_type}' not found in traces"
        return self
    
    def assert_entity_type(self, entity_type: str) -> 'TraceAssertions':
        """Assert that traces contain a specific entity type."""
        entities = self.query_engine.find_entities_by_type(entity_type)
        assert len(entities) > 0, f"Entity type '{entity_type}' not found in traces"
        return self
    
    def assert_workflow_complete(self, expected_agent_type: str = "agent.openai_agents") -> 'TraceAssertions':
        """Assert that we have a complete agent workflow."""
        complete = self.query_engine.assert_agent_workflow(expected_agent_type)
        assert complete, f"Incomplete agent workflow for type '{expected_agent_type}'"
        return self
    
    def count_llm_calls(self) -> int:
        """Count the number of LLM inference calls using JMESPath."""
        return self.query_engine.count_llm_calls()
    
    def assert_min_llm_calls(self, min_calls: int) -> 'TraceAssertions':
        """Assert minimum number of LLM calls."""
        actual_calls = self.count_llm_calls()
        assert actual_calls >= min_calls, f"Expected at least {min_calls} LLM calls, found {actual_calls}"
        return self
    
    def get_entity_names(self) -> List[str]:
        """Get all entity names from traces using JMESPath."""
        return self.query_engine.get_entity_names()
    
    def get_entity_types(self) -> List[str]:
        """Get all entity types from traces using JMESPath."""
        return self.query_engine.get_all_entity_types()
    
    def debug_entities(self) -> 'TraceAssertions':
        """Debug print all entities found in traces using JMESPath."""
        self.query_engine.debug_entities()
        return self
    
    # Direct getter methods for common queries - no need for .query_engine
    def get_agentic_spans(self) -> List[Any]:
        """Get all agentic-related spans."""
        return self.query_engine.get_agentic_spans()
    
    def get_tool_invocations(self) -> List[Any]:
        """Get all tool invocation spans."""
        return self.query_engine.get_tool_invocations()
    
    def find_entities_by_type(self, entity_type: str) -> List[Any]:
        """Find all entities of a specific type across all spans."""
        return self.query_engine.find_entities_by_type(entity_type)
    
    def find_spans_by_type(self, span_type: str) -> List[Any]:
        """Find all spans of a specific type."""
        return self.query_engine.find_spans_by_type(span_type)
    
    def get_agent_names(self) -> List[str]:
        """Get all agent names from traces using JMESPath.
        
        Returns:
            List of unique agent names found in traces
        """
        agent_names = self.query("[?attributes.\"agent.name\"].attributes.\"agent.name\"")
        return list(set(agent_names)) if agent_names else []
    
    def get_agents_by_name(self, agent_name: str) -> List[Any]:
        """Get all spans for a specific agent name using JMESPath.
        
        Args:
            agent_name: The name of the agent to find
            
        Returns:
            List of spans where the agent name matches
        """
        return self.query(f"[?attributes.\"agent.name\" == '{agent_name}']")
    
    def assert_agent_called(self, agent_name: str) -> 'TraceAssertions':
        """Assert that an agent with the given name was called.
        
        Args:
            agent_name: The name of the agent that should have been called
            
        Returns:
            TraceAssertions instance for chaining
        """
        agent_spans = self.get_agents_by_name(agent_name)
        assert len(agent_spans) > 0, f"Agent '{agent_name}' was not called. Available agents: {self.get_agent_names()}"
        return self
    
    def get_tools_used_by_agent(self, agent_name: str) -> List[str]:
        """Get all tool names used by a specific agent using JMESPath.
        
        Args:
            agent_name: The name of the agent
            
        Returns:
            List of tool names used by the agent
        """
        tool_names = self.query(f"[?attributes.\"agent.name\" == '{agent_name}'].attributes.\"tool.name\" | [?@ != null]")
        return list(set(tool_names)) if tool_names else []
    
    # Sequence and Flow Validation Methods
    def get_agent_execution_sequence(self) -> List[Dict[str, Any]]:
        """Get the chronological sequence of agent executions.
        
        Returns:
            List of agent execution info sorted by start time, including:
            - agent_name: Name of the agent
            - start_time: When the agent started
            - end_time: When the agent finished  
            - duration: Execution duration in seconds
            - span: The actual span object
        """
        agent_spans = []
        for span in self._all_spans:
            if span.attributes and span.attributes.get("agent.name"):
                agent_spans.append({
                    "agent_name": span.attributes.get("agent.name"),
                    "start_time": span.start_time,
                    "end_time": span.end_time,
                    "duration": (span.end_time - span.start_time) / 1_000_000_000,  # Convert to seconds
                    "span": span
                })
        
        # Sort by start time to get chronological sequence
        return sorted(agent_spans, key=lambda x: x["start_time"])
    
    def assert_agent_sequence(self, expected_sequence: List[str]) -> 'TraceAssertions':
        """Assert that agents were executed in a specific sequence.
        
        Args:
            expected_sequence: List of agent names in expected execution order
            
        Returns:
            TraceAssertions instance for chaining
        """
        execution_sequence = self.get_agent_execution_sequence()
        actual_sequence = [exec_info["agent_name"] for exec_info in execution_sequence]
        
        # Filter to only include agents that are in the expected sequence
        filtered_actual = [name for name in actual_sequence if name in expected_sequence]
        
        assert filtered_actual == expected_sequence, (
            f"Agent execution sequence mismatch.\n"
            f"Expected: {expected_sequence}\n"
            f"Actual: {filtered_actual}\n"
            f"Full execution order: {actual_sequence}"
        )
        return self
    
    def assert_agent_called_before(self, first_agent: str, second_agent: str) -> 'TraceAssertions':
        """Assert that one agent was called before another.
        
        Args:
            first_agent: Name of agent that should be called first
            second_agent: Name of agent that should be called second
            
        Returns:
            TraceAssertions instance for chaining
        """
        execution_sequence = self.get_agent_execution_sequence()
        
        first_agent_time = None
        second_agent_time = None
        
        for exec_info in execution_sequence:
            if exec_info["agent_name"] == first_agent and first_agent_time is None:
                first_agent_time = exec_info["start_time"]
            elif exec_info["agent_name"] == second_agent and second_agent_time is None:
                second_agent_time = exec_info["start_time"]
        
        assert first_agent_time is not None, f"Agent '{first_agent}' was not called"
        assert second_agent_time is not None, f"Agent '{second_agent}' was not called"
        assert first_agent_time < second_agent_time, (
            f"Agent '{first_agent}' was not called before '{second_agent}'. "
            f"'{first_agent}' started at {first_agent_time}, '{second_agent}' started at {second_agent_time}"
        )
        return self
    
    def assert_agents_called_in_parallel(self, agent_names: List[str], tolerance_ms: int = 1000) -> 'TraceAssertions':
        """Assert that agents were called in parallel (within a time tolerance).
        
        Args:
            agent_names: List of agent names that should execute in parallel
            tolerance_ms: Time tolerance in milliseconds for considering calls parallel
            
        Returns:
            TraceAssertions instance for chaining
        """
        execution_sequence = self.get_agent_execution_sequence()
        
        agent_start_times = {}
        for exec_info in execution_sequence:
            if exec_info["agent_name"] in agent_names:
                if exec_info["agent_name"] not in agent_start_times:
                    agent_start_times[exec_info["agent_name"]] = exec_info["start_time"]
        
        # Check that all agents were called
        missing_agents = set(agent_names) - set(agent_start_times.keys())
        assert not missing_agents, f"Agents not called: {missing_agents}"
        
        # Check that all start times are within tolerance
        start_times = list(agent_start_times.values())
        min_time = min(start_times)
        max_time = max(start_times)
        time_diff_ms = (max_time - min_time) / 1_000_000  # Convert to milliseconds
        
        assert time_diff_ms <= tolerance_ms, (
            f"Agents not called in parallel. Time difference: {time_diff_ms:.2f}ms > {tolerance_ms}ms tolerance. "
            f"Agent start times: {[(name, time) for name, time in agent_start_times.items()]}"
        )
        return self
    
    def assert_conditional_flow(self, condition_agent: str, condition_output_contains: str, 
                              then_agents: List[str], else_agents: List[str] = None) -> 'TraceAssertions':
        """Assert a conditional branching flow based on agent output.
        
        Args:
            condition_agent: Name of agent whose output determines the branch
            condition_output_contains: Text that should be in output for 'then' branch
            then_agents: Agents that should be called if condition is met
            else_agents: Agents that should be called if condition is not met (optional)
            
        Returns:
            TraceAssertions instance for chaining
        """
        from . import trace_utils
        
        # Find the condition agent's output
        condition_spans = self.get_agents_by_name(condition_agent)
        assert len(condition_spans) > 0, f"Condition agent '{condition_agent}' not found"
        
        condition_output = None
        for span in condition_spans:
            output = trace_utils.get_output_from_span(span)
            if output:
                condition_output = output
                break
        
        assert condition_output is not None, f"No output found for condition agent '{condition_agent}'"
        
        # Determine which branch should be taken
        condition_met = condition_output_contains.lower() in condition_output.lower()
        
        if condition_met:
            # Verify 'then' agents were called
            for agent in then_agents:
                self.assert_agent_called(agent)
            
            # Verify 'else' agents were NOT called (if specified)
            if else_agents:
                called_agents = self.get_agent_names()
                unexpected_agents = [agent for agent in else_agents if agent in called_agents]
                assert not unexpected_agents, (
                    f"Condition was met (output contains '{condition_output_contains}') but 'else' agents were called: {unexpected_agents}"
                )
        else:
            # Verify 'else' agents were called (if specified)
            if else_agents:
                for agent in else_agents:
                    self.assert_agent_called(agent)
                
                # Verify 'then' agents were NOT called
                called_agents = self.get_agent_names()
                unexpected_agents = [agent for agent in then_agents if agent in called_agents]
                assert not unexpected_agents, (
                    f"Condition was not met (output doesn't contain '{condition_output_contains}') but 'then' agents were called: {unexpected_agents}"
                )
        
        return self
    
    def assert_workflow_pattern(self, pattern: str, agents: List[str]) -> 'TraceAssertions':
        """Assert a specific workflow execution pattern.
        
        Args:
            pattern: The workflow pattern ('sequential', 'parallel', 'fan-out', 'fan-in')
            agents: List of agent names involved in the pattern
            
        Returns:
            TraceAssertions instance for chaining
        """
        if pattern == "sequential":
            self.assert_agent_sequence(agents)
        elif pattern == "parallel":
            self.assert_agents_called_in_parallel(agents)
        elif pattern == "fan-out":
            # First agent calls multiple others
            coordinator = agents[0]
            workers = agents[1:]
            for worker in workers:
                self.assert_agent_called_after(coordinator, worker)
        elif pattern == "fan-in":
            # Multiple agents feed into final agent
            workers = agents[:-1]
            aggregator = agents[-1]
            for worker in workers:
                self.assert_agent_called_before(worker, aggregator)
        else:
            raise ValueError(f"Unknown workflow pattern: {pattern}")
        
        return self
    
    def assert_agent_called_after(self, first_agent: str, second_agent: str) -> 'TraceAssertions':
        """Assert that one agent was called after another (alias for clarity)."""
        return self.assert_agent_called_before(first_agent, second_agent)
    
    def debug_execution_flow(self) -> 'TraceAssertions':
        """Print debug information about the execution flow for troubleshooting."""
        execution_sequence = self.get_agent_execution_sequence()
        
        print(f"\n=== EXECUTION FLOW DEBUG ({len(execution_sequence)} agent executions) ===")
        for i, exec_info in enumerate(execution_sequence):
            print(f"  [{i+1}] {exec_info['agent_name']}")
            print(f"      Start: {_format_timestamp(exec_info['start_time'])}")
            print(f"      Duration: {exec_info['duration']:.3f}s")
            
            # Show tools used
            tools = self.get_tools_used_by_agent(exec_info['agent_name'])
            if tools:
                print(f"      Tools: {tools}")
        
        print("========================\n")
        return self
    
    async def ask_llm_about_traces(self, question: str) -> str:
        """
        Ask an LLM any question about the trace data.
        
        This method:
        1. Extracts all span outputs from current traces
        2. Converts the data to structured JSON using LLM
        3. Answers the test developer's question about that data
        
        Args:
            question: Any question about the traces (e.g., "What is the total cost?", 
                     "How many API calls were made?", "What tools were used?")
                     
        Returns:
            LLM's analysis and answer to the question
        """
        try:
            import openai
            import json
            from . import trace_utils
            
            # Collect all outputs from current spans
            outputs = []
            for span in self._current_spans:
                output = trace_utils.get_output_from_span(span)
                if output:
                    outputs.append({
                        "span_name": span.name,
                        "span_type": span.attributes.get("span.type", "unknown"),
                        "output": output
                    })
            
            if not outputs:
                return "No span outputs found in current traces"
            
            # First, convert to structured data
            extraction_prompt = f"""
            Extract structured information from these agent trace outputs and return it as JSON.
            
            Trace Outputs:
            {json.dumps(outputs, indent=2)}
            
            Please analyze and structure any relevant information like:
            - Costs, prices, budgets
            - API calls made
            - Tools used
            - Errors or successes
            - Processing times
            - User requests and responses
            
            Return clean JSON with relevant extracted data.
            """
            
            client = openai.OpenAI()
            extraction_completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a data extraction assistant. Return only valid JSON."},
                    {"role": "user", "content": extraction_prompt}
                ],
                max_tokens=1000,
                temperature=0.1
            )
            
            structured_data = json.loads(extraction_completion.choices[0].message.content.strip())
            
            # Now answer the specific question
            analysis_prompt = f"""
            Answer this question based on the structured trace data:
            
            Structured Data:
            {json.dumps(structured_data, indent=2)}
            
            Question: {question}
            
            Provide a direct, specific answer. If the data doesn't contain the information needed, say "Information not available in the trace data".
            """
            
            analysis_completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that analyzes agent trace data."},
                    {"role": "user", "content": analysis_prompt}
                ],
                max_tokens=400,
                temperature=0.1
            )
            
            return analysis_completion.choices[0].message.content.strip()
            
        except ImportError:
            return "OpenAI not available for LLM analysis"
        except Exception as e:
            return f"Error in LLM analysis: {str(e)}"


class BaseAgentTest:
    """Base class for agent tests, similar to AgentiTest's BaseAgentTest."""
    
    @pytest.fixture(autouse=True)
    def setup_test(self, request):
        """Auto-used fixture to set up test context."""
        self.validator = MonocleValidator()
        self.test_name = request.node.name
        # Clear any previous spans
        self.validator.clear_spans()
            
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


# Helper functions for common assertions
def assert_agent_sequence(*agent_names: str, spans: List[ReadableSpan] = None):
    """Assert agents were called in specific sequence."""
    if spans is None:
        spans = MonocleValidator().spans
        
    agent_spans = [
        span for span in spans 
        if span.attributes.get("span.type") == "agentic_invocation"
    ]
    
    # Sort by start time to get execution order
    agent_spans.sort(key=lambda s: s.start_time)
    
    actual_sequence = [span.attributes.get("agent.name") for span in agent_spans]
    expected_sequence = list(agent_names)
    
    assert actual_sequence == expected_sequence, \
        f"Expected agent sequence {expected_sequence}, got {actual_sequence}"


def assert_tool_invocations(expected_tools: List[str], spans: List[ReadableSpan] = None):
    """Assert specific tools were invoked."""
    if spans is None:
        spans = MonocleValidator().spans
        
    tool_spans = [
        span for span in spans 
        if span.attributes.get("span.type") == "tool_invocation"
    ]
    
    actual_tools = [span.attributes.get("tool.name") for span in tool_spans]
    
    for expected_tool in expected_tools:
        assert expected_tool in actual_tools, \
            f"Expected tool '{expected_tool}' not found in invocations: {actual_tools}"


def assert_output_contains(text: str, spans: List[ReadableSpan] = None):
    """Assert any span output contains the specified text."""
    if spans is None:
        spans = MonocleValidator().spans
        
    found = False
    for span in spans:
        output = trace_utils.get_output_from_span(span)
        if output and text.lower() in output.lower():
            found = True
            break
            
    assert found, f"Text '{text}' not found in any span outputs"