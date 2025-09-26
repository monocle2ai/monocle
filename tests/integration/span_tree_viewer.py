#!/usr/bin/env python3
"""
Span Tree Viewer - Creates a hierarchical tree view of spans based on span_id and parent_id relationships.
Captures interaction between participants and agent delegations.
"""

import json
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional


class SpanNode:
    def __init__(self, span_data: Dict[str, Any]):
        self.span_data = span_data
        self.children: List['SpanNode'] = []
        self.span_id = span_data['context']['span_id']
        self.parent_id = span_data.get('parent_id')
        self.span_type = span_data['attributes'].get('span.type', 'unknown')
        self.name = span_data.get('name', 'unnamed')
        self.workflow_name = span_data['attributes'].get('workflow.name', 'unknown')
        
        # Extract agent/entity information
        self.entities = self._extract_entities()
        self.agent_info = self._extract_agent_info()
        
    def _extract_entities(self) -> List[Dict[str, str]]:
        """Extract entity information from span attributes."""
        entities = []
        attributes = self.span_data.get('attributes', {})
        
        # Look for entity.N.name, entity.N.type patterns
        entity_count = attributes.get('entity.count', 0)
        for i in range(1, entity_count + 1):
            entity = {}
            name_key = f'entity.{i}.name'
            type_key = f'entity.{i}.type'
            
            if name_key in attributes:
                entity['name'] = attributes[name_key]
            if type_key in attributes:
                entity['type'] = attributes[type_key]
            
            # Also check for other entity attributes
            for key, value in attributes.items():
                if key.startswith(f'entity.{i}.') and key not in [name_key, type_key]:
                    attr_name = key.split('.', 2)[2]  # Get the part after entity.N.
                    entity[attr_name] = value
            
            if entity:
                entities.append(entity)
                
        return entities
        
    def _extract_agent_info(self) -> Dict[str, Any]:
        """Extract agent delegation and interaction information."""
        info = {}
        
        # Check events for input/output data that might show delegation
        events = self.span_data.get('events', [])
        for event in events:
            event_name = event.get('name', '')
            if event_name in ['data.input', 'data.output']:
                attributes = event.get('attributes', {})
                if 'input' in attributes:
                    info['input'] = attributes['input']
                if 'response' in attributes:
                    info['response'] = attributes['response']
                    
        # Extract service information
        resource = self.span_data.get('resource', {})
        resource_attrs = resource.get('attributes', {})
        if 'service.name' in resource_attrs:
            info['service_name'] = resource_attrs['service.name']
        
        # Extract delegation information
        attributes = self.span_data.get('attributes', {})
        info['from_agent'] = attributes.get('entity.1.from_agent') or attributes.get('from_agent')
        info['to_agent'] = attributes.get('entity.1.to_agent') or attributes.get('to_agent')
            
        return info
    
    def get_delegation_info(self) -> str:
        """Get delegation information for display."""
        from_agent = self.agent_info.get('from_agent')
        to_agent = self.agent_info.get('to_agent')
        
        if from_agent and to_agent:
            return f"[{from_agent} -delegates-> {to_agent}]"
        
        # For agent invocations, try to get agent name from entities
        if self.span_type in ['agentic.invocation', 'agentic.request']:
            for entity in self.entities:
                if entity.get('type', '').endswith('.langgraph'):
                    agent_name = entity.get('name', 'unnamed')
                    return f"[agent: {agent_name}]"
        
        return ""
    
    def get_inference_sub_type(self) -> str:
        """Extract inference_sub_type from metadata events."""
        events = self.span_data.get('events', [])
        for event in events:
            if event.get('name') == 'metadata':
                attributes = event.get('attributes', {})
                return attributes.get('inference_sub_type', '')
        return ''
    
    def get_finish_reason(self) -> str:
        """Extract finish_reason from metadata events."""
        events = self.span_data.get('events', [])
        for event in events:
            if event.get('name') == 'metadata':
                attributes = event.get('attributes', {})
                return attributes.get('finish_reason', '')
        return ''
    
    def get_tool_call_from_response(self) -> str:
        """Extract tool call name from response data."""
        import json as json_module
        
        events = self.span_data.get('events', [])
        for event in events:
            if event.get('name') == 'data.output':
                attributes = event.get('attributes', {})
                response_str = attributes.get('response', '')
                
                if response_str:
                    try:
                        # Parse the JSON response
                        response_data = json_module.loads(response_str)
                        
                        # Check if it's a direct tool call
                        if isinstance(response_data, dict) and 'ai' in response_data:
                            ai_content = response_data['ai']
                            if isinstance(ai_content, dict) and ai_content.get('type') == 'tool_call':
                                tool_name = ai_content.get('name', '')
                                args = ai_content.get('args', {})
                                if args:
                                    # Format args nicely for common patterns
                                    if 'city' in args:
                                        return f"{tool_name}({args['city']})"
                                    elif 'from_airport' in args and 'to_airport' in args:
                                        return f"{tool_name}({args['from_airport']}→{args['to_airport']})"
                                    else:
                                        # Show first few key-value pairs
                                        arg_strs = []
                                        for k, v in list(args.items())[:2]:
                                            arg_strs.append(f"{k}:{v}")
                                        return f"{tool_name}({', '.join(arg_strs)})"
                                return tool_name
                                
                    except (json_module.JSONDecodeError, KeyError, TypeError):
                        pass
        
        return ''
    
    def get_model_name(self) -> str:
        """Get model name from entities."""
        for entity in self.entities:
            entity_type = entity.get('type', '')
            if entity_type.startswith('model.llm.'):
                return entity.get('name', 'unknown_model')
        return ''
    
    def get_tool_name(self) -> str:
        """Get tool name from entities."""
        for entity in self.entities:
            entity_type = entity.get('type', '')
            if 'tool' in entity_type or 'mcp.server' in entity_type:
                return entity.get('name', 'unknown_tool')
        return ''
    
    def get_agent_name(self) -> str:
        """Get agent name from entities."""
        for entity in self.entities:
            entity_type = entity.get('type', '')
            if 'agent' in entity_type:
                return entity.get('name', 'unknown_agent')
        return ''
    
    def get_semantic_label(self) -> str:
        """Generate semantic label based on span type and context."""
        if self.span_type == 'workflow':
            return "Multi-Agent Workflow Start"
        
        elif self.span_type == 'agentic.request':
            return "User Query Processing"
        
        elif self.span_type == 'agentic.delegation':
            from_agent = self.agent_info.get('from_agent', 'unknown')
            to_agent = self.agent_info.get('to_agent', 'unknown')
            return f"Delegation: {from_agent} → {to_agent}"
        
        elif self.span_type == 'agentic.invocation':
            agent_name = self.get_agent_name()
            return f"Agent Invocation: {agent_name}"
        
        elif self.span_type in ['inference.framework', 'inference.modelapi']:
            model_name = self.get_model_name()
            inference_sub_type = self.get_inference_sub_type()
            finish_reason = self.get_finish_reason()
            tool_call = self.get_tool_call_from_response()
            
            if inference_sub_type == 'tool_call' or finish_reason == 'tool_calls':
                if tool_call:
                    return f"LLM Decision of {model_name} → {tool_call}"
                else:
                    return f"LLM Decision of {model_name} → tool call"
            elif inference_sub_type == 'communication' or finish_reason == 'stop':
                return f"LLM Response of {model_name} → conversation"
            elif inference_sub_type == 'delegation':
                return f"LLM Decision of {model_name} → delegation"
            else:
                return f"LLM Inference: {model_name}"
        
        elif self.span_type == 'agentic.tool.invocation':
            tool_name = self.get_tool_name()
            return f"Tool Execution: {tool_name}"
        
        elif self.span_type == 'agentic.mcp.invocation':
            tool_name = self.get_tool_name()
            return f"MCP Tool: {tool_name}"
        
        elif self.span_type == 'generic':
            if 'runnable' in self.name.lower():
                return "Framework Processing"
            return "Generic Operation"
        
        else:
            return f"{self.span_type}: {self.name[:30]}..."
    
    def get_display_name(self) -> str:
        """Get the display name using semantic labels."""
        semantic_label = self.get_semantic_label()
        start_time = self.span_data.get('start_time', 'unknown')
        return f"{semantic_label} [{self.span_type}|{start_time}]"
        
    def add_child(self, child: 'SpanNode'):
        self.children.append(child)
        
    def get_duration_ms(self) -> float:
        """Calculate span duration in milliseconds."""
        try:
            start = datetime.fromisoformat(self.span_data['start_time'].replace('Z', '+00:00'))
            end = datetime.fromisoformat(self.span_data['end_time'].replace('Z', '+00:00'))
            return (end - start).total_seconds() * 1000
        except Exception:
            return 0.0


class SpanTreeBuilder:
    def __init__(self, spans_data: List[Dict[str, Any]], show_entities: bool = False):
        self.spans_data = spans_data
        self.nodes: Dict[str, SpanNode] = {}
        self.root_nodes: List[SpanNode] = []
        self.show_entities = show_entities
        
    def build_tree(self):
        """Build the tree structure from spans data."""
        # Create all nodes first
        for span_data in self.spans_data:
            span_id = span_data['context']['span_id']
            node = SpanNode(span_data)
            self.nodes[span_id] = node
            
        # Build parent-child relationships
        for node in self.nodes.values():
            if node.parent_id and node.parent_id in self.nodes:
                parent = self.nodes[node.parent_id]
                parent.add_child(node)
            else:
                # This is a root node
                self.root_nodes.append(node)
                
        # Sort root nodes by start time
        self.root_nodes.sort(key=lambda n: n.span_data.get('start_time', ''))
        
    def print_tree(self, output_file: Optional[str] = None):
        """Print the tree structure."""
        output = []
        output.append("=== SPAN TREE VIEW ===")
        output.append(f"Total spans: {len(self.nodes)}")
        output.append(f"Root spans: {len(self.root_nodes)}")
        output.append("")
        
        for root in self.root_nodes:
            self._print_node(root, "", True, output)
            
        if output_file:
            with open(output_file, 'w') as f:
                f.write('\n'.join(output))
            print(f"Tree view saved to: {output_file}")
        else:
            print('\n'.join(output))
            
    def _print_node(self, node: SpanNode, prefix: str, is_last: bool, output: List[str]):
        """Recursively print a node and its children."""
        # Create the tree structure symbols
        connector = "└── " if is_last else "├── "
        
        # Build node description using new format
        node_line = f"{prefix}{connector}{node.get_display_name()}"
        output.append(node_line)
        
        # Add details with proper indentation
        detail_prefix = prefix + ("    " if is_last else "│   ")
        duration_ms = node.get_duration_ms()
        
        output.append(f"{detail_prefix}📊 Original Name: {node.name}")
        output.append(f"{detail_prefix}🆔 Span ID: {node.span_id}")
        output.append(f"{detail_prefix}⏱️  Duration: {duration_ms:.2f}ms")
        output.append(f"{detail_prefix}🔧 Workflow: {node.workflow_name}")
            
        # Show entities (tools, models, agents, etc.) - controlled by flag
        if self.show_entities and node.entities:
            output.append(f"{detail_prefix}🎯 Entities:")
            for entity in node.entities:
                entity_type = entity.get('type', 'unknown')
                entity_name = entity.get('name', 'unnamed')
                output.append(f"{detail_prefix}   • {entity_name} ({entity_type})")
                
                # Show additional entity attributes
                for key, value in entity.items():
                    if key not in ['name', 'type']:
                        output.append(f"{detail_prefix}     - {key}: {value}")
        
        # Show agent delegation info for specific span types
        if node.span_type in ['agentic.delegation', 'agentic.invocation', 'agentic.tool.invocation']:
            output.append(f"{detail_prefix}🤖 Agent Interaction:")
            
            # Show input/output if available
            if 'input' in node.agent_info:
                input_data = node.agent_info['input']
                if isinstance(input_data, list):
                    for i, inp in enumerate(input_data):
                        output.append(f"{detail_prefix}   📥 Input {i+1}: {inp[:100]}..." if len(str(inp)) > 100 else f"{detail_prefix}   📥 Input {i+1}: {inp}")
                else:
                    input_str = str(input_data)
                    output.append(f"{detail_prefix}   📥 Input: {input_str[:100]}..." if len(input_str) > 100 else f"{detail_prefix}   📥 Input: {input_str}")
                    
            if 'response' in node.agent_info:
                response = str(node.agent_info['response'])
                output.append(f"{detail_prefix}   📤 Response: {response[:100]}..." if len(response) > 100 else f"{detail_prefix}   📤 Response: {response}")
        
        # Show events for inference spans
        if node.span_type.startswith('inference.') and node.span_data.get('events'):
            events = node.span_data['events']
            metadata_events = [e for e in events if e.get('name') == 'metadata']
            if metadata_events:
                metadata = metadata_events[0].get('attributes', {})
                if metadata:
                    output.append(f"{detail_prefix}📈 Inference Metrics:")
                    for key, value in metadata.items():
                        if key in ['finish_reason']:  # Hide token information, only show finish_reason
                            output.append(f"{detail_prefix}   • {key}: {value}")
                            
        output.append("")
        
        # Print children (filter out inference.modelapi spans)
        visible_children = [child for child in node.children if child.span_type != 'inference.modelapi']
        for i, child in enumerate(visible_children):
            is_child_last = i == len(visible_children) - 1
            child_prefix = prefix + ("    " if is_last else "│   ")
            self._print_node(child, child_prefix, is_child_last, output)


def main():
    """Main function to process the spans file."""
    if len(sys.argv) < 2:
        print("Usage: python span_tree_viewer.py <spans_json_file> [--show-entities]")
        print("  --show-entities: Show entity information in the display (hidden by default)")
        sys.exit(1)
        
    input_file = sys.argv[1]
    show_entities = False  # Hide entities by default
    
    # Parse additional flags
    if len(sys.argv) > 2:
        for arg in sys.argv[2:]:
            if arg == '--show-entities':
                show_entities = True
            else:
                print(f"Unknown argument: {arg}")
                print("Usage: python span_tree_viewer.py <spans_json_file> [--show-entities]")
                sys.exit(1)
    
    try:
        with open(input_file, 'r') as f:
            spans_data = json.load(f)
            
        if not isinstance(spans_data, list):
            print("Error: JSON file should contain a list of spans")
            sys.exit(1)
            
        print(f"Loading {len(spans_data)} spans from {input_file}")
        
        # Build and print the tree
        builder = SpanTreeBuilder(spans_data, show_entities=show_entities)
        builder.build_tree()
        
        # Generate output filename
        output_file = input_file.replace('.json', '_tree_view.txt')
        builder.print_tree(output_file)
        
    except FileNotFoundError:
        print(f"Error: File {input_file} not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file {input_file}: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()