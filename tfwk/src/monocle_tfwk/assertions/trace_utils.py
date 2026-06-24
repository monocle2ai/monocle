import logging
from typing import Any, Dict, List

import jmespath
from opentelemetry.sdk.trace import Span

logger = logging.getLogger(__name__)


def get_input_from_span(span: Span) -> str:
    """
    Extracts the input text from the span attributes.
    
    Args:
        span (Span): The span object from which to extract the input.
        
    Returns:
        str: The extracted input text, or an empty string if not found.
    """
    # First check span attributes directly
    if span.attributes and "input" in span.attributes:
        return span.attributes.get("input", "")
    
    # Then check events
    for event in span.events:
        if event.name == "data.input":
            return event.attributes.get("input", "")
    return None

def get_output_from_span(span: Span) -> str:
    """
    Extracts the output text from the span attributes.

    Args:
        span (Span): The span object from which to extract the output.

    Returns:
        str: The extracted output text, or an empty string if not found.
    """
    # First check span attributes directly
    if span.attributes and "output" in span.attributes:
        return span.attributes.get("output", "")
    
    # Then check events
    for event in span.events:
        if event.name == "data.output":
            return event.attributes.get("response")
    return None




class TraceQueryEngine:
    """JMESPath-based query engine for extracting data from Monocle traces."""
    
    def __init__(self, traces):
        """Initialize with trace data."""
        self.traces = traces
        # Convert spans to a queryable format
        self.span_data = []
        for span in traces.spans:
            # Extract span ID properly - handle both ReadableSpan and dict formats
            span_id = None
            if hasattr(span, 'context') and hasattr(span.context, 'span_id') and span.context.span_id:
                span_id = span.context.span_id.to_bytes(8, 'big').hex()
            elif hasattr(span, 'span_id'):
                span_id = str(span.span_id)
            
            span_dict = {
                'name': span.name,
                'span_id': span_id,
                'attributes': dict(span.attributes) if hasattr(span, 'attributes') and span.attributes else {},
                'start_time': span.start_time if hasattr(span, 'start_time') else None,
                'end_time': span.end_time if hasattr(span, 'end_time') else None,
                'status': str(span.status) if hasattr(span, 'status') else None
            }
            self.span_data.append(span_dict)
    
    def query(self, jmes_expression: str) -> Any:
        """Execute a JMESPath query on the trace data."""
        try:
            return jmespath.search(jmes_expression, self.span_data)
        except Exception as e:
            logger.error(f"JMESPath query failed: {e}")
            return None
    
    def find_entities_by_type(self, entity_type: str) -> List[Dict[str, Any]]:
        """Find all entities of a specific type across all spans."""
        query = f"[?attributes.\"entity.1.type\" == '{entity_type}' || attributes.\"entity.2.type\" == '{entity_type}' || attributes.\"entity.3.type\" == '{entity_type}']"
        return self.query(query) or []
    
    def find_spans_by_type(self, span_type: str) -> List[Dict[str, Any]]:
        """Find all spans of a specific type."""
        query = f"[?attributes.\"span.type\" == '{span_type}']"
        return self.query(query) or []
    
    def get_entity_names(self) -> List[str]:
        """Get all entity names from the traces."""
        query = "[].attributes.\"entity.1.name\" | [?@ != null]"
        entity1_names = self.query(query) or []
        query = "[].attributes.\"entity.2.name\" | [?@ != null]"
        entity2_names = self.query(query) or []
        return list(set(entity1_names + entity2_names))
    
    def get_all_entity_types(self) -> List[str]:
        """Get all entity types from the traces."""
        query = "[].attributes.\"entity.1.type\" | [?@ != null]"
        entity1_types = self.query(query) or []
        query = "[].attributes.\"entity.2.type\" | [?@ != null]"
        entity2_types = self.query(query) or []
        query = "[].attributes.\"entity.3.type\" | [?@ != null]"
        entity3_types = self.query(query) or []
        return list(set(entity1_types + entity2_types + entity3_types))
    
    def has_agent_type(self, agent_type: str) -> bool:
        """Check if traces contain a specific agent type."""
        agents = self.find_entities_by_type(agent_type)
        return len(agents) > 0
    
    def get_agentic_spans(self) -> List[Dict[str, Any]]:
        """Get all agentic-related spans."""
        query = "[?contains(attributes.\"span.type\", 'agentic') || contains(name, 'agent')]"
        return self.query(query) or []
    
    def get_tool_invocations(self) -> List[Dict[str, Any]]:
        """Get all tool invocation spans."""
        query = "[?attributes.\"span.type\" == 'agentic.tool.invocation' || contains(name, 'tool')]"
        return self.query(query) or []
    
    def count_llm_calls(self) -> int:
        """Count the number of LLM inference calls."""
        query = "length([?attributes.\"span.type\" == 'inference' || attributes.\"entity.1.type\" == 'inference.openai'])"
        return self.query(query) or 0
    
    def assert_agent_workflow(self, expected_agent_type: str = "agent.openai_agents") -> bool:
        """Assert that we have a complete agent workflow with the expected agent type."""
        # Check for agent spans
        agent_spans = self.find_entities_by_type(expected_agent_type)
        if not agent_spans:
            return False
        
        # Check for agentic spans
        agentic_spans = self.get_agentic_spans()
        if not agentic_spans:
            return False
        
        return True
    
    def debug_entities(self):
        """Debug print all entities found in traces."""
        print("\n🔍 JMESPath Entity Debug:")
        print(f"Entity Types: {self.get_all_entity_types()}")
        print(f"Entity Names: {self.get_entity_names()}")
        print(f"Agent Spans: {len(self.get_agentic_spans())}")
        print(f"Tool Invocations: {len(self.get_tool_invocations())}")
        print(f"LLM Calls: {self.count_llm_calls()}")
        
        # Show first few spans with their attributes
        for i, span in enumerate(self.span_data[:3]):
            print(f"Span {i}: {span['name']}")
            for key, value in span['attributes'].items():
                if key.startswith('entity.') or key.startswith('span.'):
                    print(f"  {key}: {value}")
        print("=" * 50)