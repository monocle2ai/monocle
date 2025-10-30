"""
Fluent assertion API for agent traces with JMESPath query capabilities.

This module provides the TraceAssertions class for validating agent execution traces.
"""
import json
import logging
from datetime import datetime
from typing import Any, Dict, List

from opentelemetry.sdk.trace import ReadableSpan

from monocle_tfwk.assertions import trace_utils
from monocle_tfwk.assertions.plugin_registry import TraceAssertionsPluginRegistry

logger = logging.getLogger(__name__)


def _format_timestamp(nanoseconds: int) -> str:
    """Convert nanosecond timestamp to ISO 8601 format for readable output."""
    seconds = nanoseconds / 1_000_000_000
    dt = datetime.fromtimestamp(seconds)
    return dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')


class TraceAssertions:
    """
    Fluent assertion API for agent traces with JMESPath query capabilities.
    
    This class supports extensibility through plugins that can add custom assertion methods.
    Plugins are automatically loaded from the TraceAssertionsPluginRegistry when instances
    are created.
    
    ## Plugin System
    
    TraceAssertions can be extended with custom assertion methods through plugins.
    Plugins inherit from TraceAssertionsPlugin and are registered using the @plugin decorator
    or by calling TraceAssertions.register_plugin().
    
    Example plugin:
        ```python
        from monocle_tfwk.trace_assertions_plugins import TraceAssertionsPlugin, plugin
        
        @plugin
        class MyCustomPlugin(TraceAssertionsPlugin):
            @classmethod
            def get_plugin_name(cls) -> str:
                return "my_custom"
            
            def assert_custom_condition(self, value: str) -> 'TraceAssertions':
                '''Assert some custom condition.'''
                # Access current spans via self._current_spans
                # Access all spans via self._all_spans
                # Perform your custom assertions
                assert some_condition, "Custom assertion failed"
                return self  # Enable method chaining
        ```
    
    Plugin methods are automatically available on all TraceAssertions instances:
        ```python
        assertions = TraceAssertions(spans)
        assertions.assert_custom_condition("test_value")
        ```
    
    ## Method Chaining
    
    All assertion methods return the TraceAssertions instance to enable fluent chaining:
        ```python
        (assertions
         .assert_spans(min_count=1)
         .output_contains("result")
         .completed_successfully())
        ```
    
    ## Built-in Plugin Categories
    
    The framework includes several built-in plugin categories:
    - Performance assertions (timing, duration limits)
    - Cost assertions (token usage, API costs)
    - Security assertions (sensitive data detection)
    - Business logic assertions (domain-specific validations)
    
    See `example_plugins.py` for complete implementations.
    """
    
    def __init__(self, spans: List[ReadableSpan]):
        self._all_spans = spans
        self._current_spans = spans
        self._query_engine = None
        
        # Load plugin methods dynamically
        self._load_plugin_methods()
    
    def _load_plugin_methods(self) -> None:
        """Dynamically load methods from registered plugins."""
        plugin_methods = TraceAssertionsPluginRegistry.get_available_methods()
        
        for method_name, method in plugin_methods.items():
            # Bind the method to this instance
            bound_method = method.__get__(self, TraceAssertions)
            setattr(self, method_name, bound_method)
    
    @classmethod
    def register_plugin(cls, plugin_class) -> None:
        """
        Register a plugin class to extend TraceAssertions capabilities.
        
        Args:
            plugin_class: Plugin class inheriting from TraceAssertionsPlugin
            
        Example:
            @TraceAssertions.register_plugin
            class MyPlugin(TraceAssertionsPlugin):
                @classmethod
                def get_plugin_name(cls) -> str:
                    return "my_plugin"
                
                def assert_custom(self, value: str) -> 'TraceAssertions':
                    # Custom assertion logic
                    return self
        """
        TraceAssertionsPluginRegistry.register_plugin(plugin_class)
    
    @classmethod  
    def unregister_plugin(cls, plugin_name: str) -> None:
        """Unregister a plugin by name."""
        TraceAssertionsPluginRegistry.unregister_plugin(plugin_name)
    
    @classmethod
    def list_plugins(cls) -> Dict[str, type]:
        """List all registered plugins."""
        return TraceAssertionsPluginRegistry.get_registered_plugins()
    
    def reload_plugins(self) -> None:
        """Reload plugin methods (useful after registering new plugins)."""
        self._load_plugin_methods()
        
    # ========================
    # PROPERTIES & UTILITIES
    # ========================
        
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
        
    def count(self) -> int:
        """Return the count of current matching spans."""
        return len(self._current_spans)
        
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
    
    def debug_entities(self) -> 'TraceAssertions':
        """Debug print all entities found in traces using JMESPath."""
        self.query_engine.debug_entities()
        return self
    
    # ==============================
    # QUERY & FILTERING METHODS
    # ==============================
    
    def query(self, jmes_expression: str) -> Any:
        """Execute a JMESPath query on the trace data.
        
        Args:
            jmes_expression: JMESPath query expression
            
        Returns:
            Query result
        """
        return self.query_engine.query(jmes_expression)
        
    def filter_by_name(self, name: str) -> 'TraceAssertions':
        """Filter spans by name."""
        matching_spans = [
            span for span in self._current_spans
            if span.name == name
        ]
        self._current_spans = matching_spans
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
        
    def llm_calls(self) -> 'TraceAssertions':
        """Filter to only LLM call spans."""
        llm_spans = []
        for span in self._current_spans:
            is_llm_call = False
            

            # OpenAI inference detection
            if (span.attributes.get("span.type") == "inference" and 
                  span.attributes.get("entity.1.type") == "inference.openai"):
                is_llm_call = True
            elif any(key.startswith("entity.") and key.endswith(".type") and 
                    "model.llm" in str(span.attributes.get(key, ""))
                    for key in span.attributes.keys() if span.attributes):
                is_llm_call = True
                
            if is_llm_call:
                llm_spans.append(span)
                
        self._current_spans = llm_spans
        return self
    
    # ===========================
    # DATA RETRIEVAL METHODS
    # ===========================
    
    def get_agent_names(self) -> List[str]:
        """Get all agent names from traces using JMESPath.
        
        Returns:
            List of unique agent names found in traces
        """
        # Try multiple patterns for different agent frameworks
        agent_names = []
        
        # OpenAI agents pattern
        openai_names = self.query("[?attributes.\"entity.1.name\" && attributes.\"entity.1.type\" == 'agent.openai_agents'].attributes.\"entity.1.name\"")
        if openai_names:
            agent_names.extend(openai_names)
        
        # Traditional agent pattern
        traditional_names = self.query("[?attributes.\"agent.name\"].attributes.\"agent.name\"")
        if traditional_names:
            agent_names.extend(traditional_names)
            
        return list(set(agent_names)) if agent_names else []
    
    def get_agents_by_name(self, agent_name: str) -> List[Any]:
        """Get all spans for a specific agent name using JMESPath.
        
        Args:
            agent_name: The name of the agent to find
            
        Returns:
            List of spans where the agent name matches
        """
        return self.query(f"[?attributes.\"agent.name\" == '{agent_name}']")
    
    def get_tools_used_by_agent(self, agent_name: str) -> List[str]:
        """Get all tool names used by a specific agent using JMESPath.
        
        Args:
            agent_name: The name of the agent
            
        Returns:
            List of tool names used by the agent
        """
        tool_names = self.query(f"[?attributes.\"agent.name\" == '{agent_name}'].attributes.\"tool.name\" | [?@ != null]")
        return list(set(tool_names)) if tool_names else []
    
    def get_entity_names(self) -> List[str]:
        """Get all entity names from traces using JMESPath."""
        return self.query_engine.get_entity_names()
    
    def get_entity_types(self) -> List[str]:
        """Get all entity types from traces using JMESPath."""
        return self.query_engine.get_all_entity_types()
    
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
            agent_name = None
            if span.attributes:
                # Check for OpenAI agents pattern
                if (span.attributes.get("entity.1.type") == "agent.openai_agents" and 
                    span.attributes.get("entity.1.name")):
                    agent_name = span.attributes.get("entity.1.name")
                # Check for traditional agent pattern
                elif span.attributes.get("agent.name"):
                    agent_name = span.attributes.get("agent.name")
                
                if agent_name:
                    agent_spans.append({
                        "agent_name": agent_name,
                        "start_time": span.start_time,
                        "end_time": span.end_time,
                        "duration": (span.end_time - span.start_time) / 1_000_000_000,  # Convert to seconds
                        "span": span
                    })
        
        # Sort by start time to get chronological sequence
        return sorted(agent_spans, key=lambda x: x["start_time"])
    
    def count_llm_calls(self) -> int:
        """Count the number of LLM inference calls using JMESPath."""
        return self.query_engine.count_llm_calls()
    
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
            # Now answer the specific question
            analysis_prompt = f"""
            Answer this question based on the trace data:

            Trace Data:
            {extraction_prompt}
            
            Question: {question}
            
            Provide a direct, specific answer. If the data doesn't contain the information needed, say "Information not available in the trace data".
            """
            
            analysis_completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that analyzes agent trace data. Always respond with valid JSON."},
                    {"role": "user", "content": analysis_prompt}
                ],
                max_tokens=4000,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            return analysis_completion.choices[0].message.content.strip()
            
        except ImportError:
            return "OpenAI not available for LLM analysis"
        except Exception as e:
            return f"Error in LLM analysis: {str(e)}"
  