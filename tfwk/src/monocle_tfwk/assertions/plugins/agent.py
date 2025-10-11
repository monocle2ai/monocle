"""
Agent-specific assertion plugins for agent workflow and execution validation.

This module contains plugins that provide assertions specifically for validating
agent behavior, tool usage, and multi-agent workflows.
"""
from typing import TYPE_CHECKING, List

from monocle_tfwk.assertions import trace_utils
from monocle_tfwk.assertions.plugin_registry import TraceAssertionsPlugin, plugin

if TYPE_CHECKING:
    from monocle_tfwk.assertions.trace_assertions import TraceAssertions


@plugin
class AgentAssertionsPlugin(TraceAssertionsPlugin):
    """Plugin providing agent-specific assertion methods."""
    
    @classmethod
    def get_plugin_name(cls) -> str:
        return "agent"
    
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
    
    def assert_agent_type(self, agent_type: str) -> 'TraceAssertions':
        """Assert that traces contain a specific agent type using JMESPath."""
        found = self.query_engine.has_agent_type(agent_type)
        assert found, f"Agent type '{agent_type}' not found in traces"
        return self
    
    def assert_agent_called(self, agent_name: str) -> 'TraceAssertions':
        """Assert that an agent with the given name was called."""
        agent_spans = self.get_agents_by_name(agent_name)
        assert len(agent_spans) > 0, f"Agent '{agent_name}' was not called. Available agents: {self.get_agent_names()}"
        return self


@plugin
class AgentWorkflowAssertionsPlugin(TraceAssertionsPlugin):
    """Plugin providing complex agent workflow validation."""
    
    @classmethod
    def get_plugin_name(cls) -> str:
        return "agent_workflow"
    
    def assert_agent_sequence(self, expected_sequence: List[str]) -> 'TraceAssertions':
        """Assert that agents were executed in a specific sequence."""
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
        """Assert that one agent was called before another."""
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
        """Assert that agents were called in parallel (within a time tolerance)."""
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
        """Assert a conditional branching flow based on agent output."""
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
        """Assert a specific workflow execution pattern."""
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