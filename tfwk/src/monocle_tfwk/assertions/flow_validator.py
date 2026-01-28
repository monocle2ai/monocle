"""
Flow validation for Monocle traces.

This module provides tools for expressing expected execution patterns
and validating them against actual trace data to ensure applications
behave as expected.
"""
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union


@dataclass
class TimelineEvent:
    """Represents a single event in the trace timeline."""
    span_id: str
    parent_id: Optional[str]
    trace_id: str
    name: str
    start_time: datetime
    end_time: datetime
    duration_ms: float
    level: int = 0  # Hierarchical level for visualization
    attributes: Dict[str, Any] = field(default_factory=dict)
    children: List['TimelineEvent'] = field(default_factory=list)
    
    @property
    def span_type(self) -> str:
        """Get the span type from attributes."""
        return self.attributes.get("span.type", "unknown")
    
    @property
    def workflow_name(self) -> str:
        """Get the workflow name from attributes."""
        return self.attributes.get("workflow.name", "unknown")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "trace_id": self.trace_id,
            "name": self.name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_ms": self.duration_ms,
            "level": self.level,
            "span_type": self.span_type,
            "workflow_name": self.workflow_name,
            "attributes": self.attributes,
            "children": [child.to_dict() for child in self.children]
        }
    
class FlowOperator(Enum):
    """Operators for flow pattern matching."""
    SEQUENCE = "->"      # A follows B
    PARALLEL = "||"      # A and B happen in parallel  
    OPTIONAL = "?"       # A is optional
    REPEAT = "*"         # A repeats 0 or more times
    REPEAT_ONCE = "+"    # A repeats 1 or more times
    CHOICE = "|"         # Either A or B


@dataclass
class FlowPattern:
    """Represents an expected flow pattern."""
    name: str
    pattern: str
    description: str = ""
    
    def __post_init__(self):
        """Parse the pattern after initialization."""
        self.parsed_pattern = self._parse_pattern()
    
    def _parse_pattern(self) -> Dict[str, Any]:
        """Parse the pattern string into a structured format with support for nested expressions."""
        try:
            # Use enhanced parser that supports parentheses and nesting
            parsed_ast = self._parse_expression(self.pattern)
            
            # Extract flat lists for backwards compatibility with existing validation logic
            steps, operators = self._flatten_ast(parsed_ast)
            
            return {
                "steps": steps,
                "operators": operators,
                "raw_pattern": self.pattern,
                "ast": parsed_ast  # Include AST for future enhanced validation
            }
        except Exception:
            # Fallback to simple parsing if enhanced parsing fails
            return self._parse_pattern_simple()
    
    def _parse_expression(self, pattern: str) -> Dict[str, Any]:
        """Parse pattern into an Abstract Syntax Tree (AST) supporting nested expressions."""
        tokens = self._tokenize(pattern)
        parser = ExpressionParser(tokens)
        return parser.parse()
    
    def _tokenize(self, pattern: str) -> List[Dict[str, str]]:
        """Tokenize pattern string into a list of tokens."""
        tokens = []
        i = 0
        current_token = ""
        
        while i < len(pattern):
            char = pattern[i]
            
            if char.isspace():
                if current_token.strip():
                    tokens.append({"type": "identifier", "value": current_token.strip()})
                    current_token = ""
            elif char == '(':
                if current_token.strip():
                    tokens.append({"type": "identifier", "value": current_token.strip()})
                    current_token = ""
                tokens.append({"type": "lparen", "value": "("})
            elif char == ')':
                if current_token.strip():
                    tokens.append({"type": "identifier", "value": current_token.strip()})
                    current_token = ""
                tokens.append({"type": "rparen", "value": ")"})
            elif char == '-' and i + 1 < len(pattern) and pattern[i + 1] == '>':
                if current_token.strip():
                    tokens.append({"type": "identifier", "value": current_token.strip()})
                    current_token = ""
                tokens.append({"type": "sequence", "value": "->"})
                i += 1  # Skip the '>'
            elif char == '|' and i + 1 < len(pattern) and pattern[i + 1] == '|':
                if current_token.strip():
                    tokens.append({"type": "identifier", "value": current_token.strip()})
                    current_token = ""
                tokens.append({"type": "parallel", "value": "||"})
                i += 1  # Skip the second '|'
            elif char in ['?', '*', '+', '|']:
                if current_token.strip():
                    tokens.append({"type": "identifier", "value": current_token.strip()})
                    current_token = ""
                token_type = {"?": "optional", "*": "wildcard", "+": "repeat", "|": "choice"}[char]
                tokens.append({"type": token_type, "value": char})
            else:
                current_token += char
            
            i += 1
        
        # Add final token
        if current_token.strip():
            tokens.append({"type": "identifier", "value": current_token.strip()})
        
        return tokens
    
    def _flatten_ast(self, ast: Dict[str, Any]) -> tuple[List[str], List[str]]:
        """Flatten AST back to simple lists for backwards compatibility."""
        steps = []
        operators = []
        
        def extract_from_node(node):
            if node["type"] == "identifier":
                steps.append(node["value"])
            elif node["type"] in ["sequence", "parallel", "choice"]:
                if "left" in node and "right" in node:
                    extract_from_node(node["left"])
                    operators.append(node["value"])
                    extract_from_node(node["right"])
            elif node["type"] == "group":
                extract_from_node(node["expression"])
            elif node["type"] in ["optional", "wildcard", "repeat"]:
                extract_from_node(node["operand"])
                operators.append(node["value"])
        
        extract_from_node(ast)
        return steps, operators
    
    def _parse_pattern_simple(self) -> Dict[str, Any]:
        """Fallback simple pattern parsing for backwards compatibility."""
        steps = []
        current_step = ""
        operators = []
        
        i = 0
        while i < len(self.pattern):
            char = self.pattern[i]
            
            if char in ['-', '>', '|', '?', '*', '+']:
                if current_step.strip():
                    steps.append(current_step.strip())
                    current_step = ""
                
                # Handle multi-character operators
                if char == '-' and i + 1 < len(self.pattern) and self.pattern[i + 1] == '>':
                    operators.append("->")
                    i += 1
                elif char == '|' and i + 1 < len(self.pattern) and self.pattern[i + 1] == '|':
                    operators.append("||")
                    i += 1
                else:
                    operators.append(char)
            else:
                current_step += char
                
            i += 1
        
        # Add the last step
        if current_step.strip():
            steps.append(current_step.strip())
            
        return {
            "steps": steps,
            "operators": operators,
            "raw_pattern": self.pattern
        }


class ExpressionParser:
    """Parser for flow pattern expressions with support for nested parentheses."""
    
    def __init__(self, tokens: List[Dict[str, str]]):
        self.tokens = tokens
        self.pos = 0
    
    def parse(self) -> Dict[str, Any]:
        """Parse tokens into an AST."""
        if not self.tokens:
            return {"type": "empty", "value": ""}
        
        expr = self._parse_expression()
        if self.pos < len(self.tokens):
            # There are unparsed tokens - this might be an error or complex expression
            pass
        return expr
    
    def _current_token(self) -> Dict[str, str]:
        """Get current token without advancing position."""
        if self.pos >= len(self.tokens):
            return {"type": "eof", "value": ""}
        return self.tokens[self.pos]
    
    def _advance(self) -> Dict[str, str]:
        """Advance to next token and return current one."""
        token = self._current_token()
        self.pos += 1
        return token
    
    def _parse_expression(self) -> Dict[str, Any]:
        """Parse a full expression with operator precedence."""
        left = self._parse_sequence()
        
        while self._current_token()["type"] == "choice":
            op_token = self._advance()
            right = self._parse_sequence()
            left = {
                "type": "choice",
                "value": op_token["value"],
                "left": left,
                "right": right
            }
        
        return left
    
    def _parse_sequence(self) -> Dict[str, Any]:
        """Parse sequence expressions (-> and ||)."""
        left = self._parse_term()
        
        while self._current_token()["type"] in ["sequence", "parallel"]:
            op_token = self._advance()
            right = self._parse_term()
            left = {
                "type": op_token["type"],
                "value": op_token["value"],
                "left": left,
                "right": right
            }
        
        return left
    
    def _parse_term(self) -> Dict[str, Any]:
        """Parse a single term (identifier, group, or modified term)."""
        token = self._current_token()
        
        if token["type"] == "lparen":
            # Parse grouped expression
            self._advance()  # consume '('
            expr = self._parse_expression()
            if self._current_token()["type"] == "rparen":
                self._advance()  # consume ')'
            group = {
                "type": "group",
                "expression": expr
            }
            return self._parse_modifiers(group)
        
        elif token["type"] == "identifier":
            self._advance()
            node = {
                "type": "identifier",
                "value": token["value"]
            }
            return self._parse_modifiers(node)
        
        else:
            # Unexpected token - return as identifier for robustness
            self._advance()
            return {
                "type": "identifier", 
                "value": token["value"]
            }
    
    def _parse_modifiers(self, node: Dict[str, Any]) -> Dict[str, Any]:
        """Parse optional modifiers (?, *, +) that apply to the node."""
        while self._current_token()["type"] in ["optional", "wildcard", "repeat"]:
            mod_token = self._advance()
            node = {
                "type": mod_token["type"],
                "value": mod_token["value"],
                "operand": node
            }
        return node


class FlowValidator:
    """Validates trace execution against expected flow patterns."""
    
    def __init__(self, timeline_events: List[TimelineEvent]):
        """
        Initialize with timeline events.
        
        Args:
            timeline_events: List of TimelineEvent objects
        """
        self.timeline_events = timeline_events
    
    @classmethod
    def from_spans(cls, spans: List) -> 'FlowValidator':
        """
        Create a FlowValidator directly from ReadableSpan objects.
        
        Args:
            spans: List of ReadableSpan objects from OpenTelemetry
            
        Returns:
            FlowValidator instance
        """
        timeline_events = cls.create_timeline_events_from_spans(spans)
        return cls(timeline_events=timeline_events)
    
    @staticmethod
    def create_timeline_events_from_spans(spans: List) -> List[TimelineEvent]:
        """
        Convert ReadableSpan objects to TimelineEvent objects.
        
        Args:
            spans: List of ReadableSpan objects from OpenTelemetry
            
        Returns:
            List of TimelineEvent objects
        """
        timeline_events = []
        for span in spans:
            start_time = datetime.fromtimestamp(span.start_time / 1e9) if span.start_time else datetime.now()
            end_time = datetime.fromtimestamp(span.end_time / 1e9) if span.end_time else datetime.now()
            duration_ms = (span.end_time - span.start_time) / 1e6 if (span.end_time and span.start_time) else 0.0
            
            # Extract attributes safely
            attributes = dict(span.attributes) if span.attributes else {}
            
            event = TimelineEvent(
                span_id=span.context.span_id.to_bytes(8, 'big').hex() if span.context and span.context.span_id else "unknown",
                parent_id=span.parent.span_id.to_bytes(8, 'big').hex() if (span.parent and span.parent.span_id) else None,
                trace_id=span.context.trace_id.to_bytes(16, 'big').hex() if span.context and span.context.trace_id else "unknown",
                name=span.name,
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                attributes=attributes
            )
            timeline_events.append(event)
        
        return timeline_events
        
    def validate_pattern(self, pattern: Union[str, FlowPattern]) -> Dict[str, Any]:
        """
        Validate a flow pattern against the trace data.
        
        Args:
            pattern: Flow pattern string or FlowPattern object
            
        Returns:
            Validation result with matches, violations, and suggestions
        """
        if isinstance(pattern, str):
            pattern = FlowPattern("adhoc", pattern)
            
        result = {
            "pattern_name": pattern.name,
            "pattern": pattern.pattern,
            "valid": False,
            "matches": [],
            "violations": [],
            "suggestions": [],
            "statistics": {}
        }
        
        # Parse pattern and validate
        parsed = pattern.parsed_pattern
        validation_result = self._validate_parsed_pattern(parsed)
        
        result.update(validation_result)
        result["valid"] = len(result["violations"]) == 0
        
        return result
    
    def _validate_parsed_pattern(self, parsed_pattern: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a parsed pattern against timeline events."""
        # Check if we have AST-based validation available
        if "ast" in parsed_pattern and parsed_pattern["ast"]:
            return self._validate_ast_pattern(parsed_pattern["ast"])
        
        # Fallback to legacy validation
        return self._validate_legacy_pattern(parsed_pattern)
    
    def _validate_ast_pattern(self, ast: Dict[str, Any]) -> Dict[str, Any]:
        """Validate pattern using AST structure for enhanced grouping support."""
        matches = []
        violations = []
        suggestions = []
        
        try:
            result = self._validate_ast_node(ast)
            matches.extend(result.get("matches", []))
            violations.extend(result.get("violations", []))
            suggestions.extend(result.get("suggestions", []))
            
        except Exception as e:
            violations.append(f"AST validation failed: {str(e)}")
            suggestions.append("Pattern may contain unsupported syntax. Try using simpler expressions.")
        
        return {
            "matches": matches,
            "violations": violations,
            "suggestions": suggestions
        }
    
    def _validate_ast_node(self, node: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively validate AST nodes."""
        matches = []
        violations = []
        suggestions = []
        
        node_type = node.get("type", "")
        
        if node_type == "identifier":
            # Validate single identifier
            step_name = node["value"]
            matching_spans = self._find_matching_spans(step_name)
            
            if matching_spans:
                matches.append(f"Found spans matching '{step_name}': {len(matching_spans)} spans")
            else:
                violations.append(f"No spans found matching step: '{step_name}'")
                available_names = list(set(e.name for e in self.timeline_events))[:5]
                suggestions.append(f"Available span names: {available_names}")
        
        elif node_type == "sequence":
            # Validate A -> B (sequential execution)
            left_result = self._validate_ast_node(node["left"])
            right_result = self._validate_ast_node(node["right"])
            
            # Combine results
            matches.extend(left_result["matches"])
            matches.extend(right_result["matches"])
            violations.extend(left_result["violations"])
            violations.extend(right_result["violations"])
            
            # If both sides are valid, validate timing relationship
            if not left_result["violations"] and not right_result["violations"]:
                left_spans = self._get_spans_from_ast_node(node["left"])
                right_spans = self._get_spans_from_ast_node(node["right"])
                
                sequence_check = self._validate_sequential_order(left_spans, right_spans)
                if sequence_check["valid"]:
                    matches.append(f"Sequential order validated: {node['left'].get('value', 'left')} -> {node['right'].get('value', 'right')}")
                else:
                    violations.extend(sequence_check["violations"])
        
        elif node_type == "parallel":
            # Validate A || B (parallel execution)
            left_result = self._validate_ast_node(node["left"])
            right_result = self._validate_ast_node(node["right"])
            
            # Combine results
            matches.extend(left_result["matches"])
            matches.extend(right_result["matches"])
            violations.extend(left_result["violations"])
            violations.extend(right_result["violations"])
            
            # If both sides are valid, validate parallel timing
            if not left_result["violations"] and not right_result["violations"]:
                left_spans = self._get_spans_from_ast_node(node["left"])
                right_spans = self._get_spans_from_ast_node(node["right"])
                
                parallel_check = self._validate_parallel_execution(left_spans, right_spans)
                if parallel_check["valid"]:
                    matches.append(f"Parallel execution validated: {node['left'].get('value', 'left')} || {node['right'].get('value', 'right')}")
                else:
                    violations.extend(parallel_check["violations"])
        
        elif node_type == "group":
            # Validate grouped expression (A || B) 
            expr_result = self._validate_ast_node(node["expression"])
            matches.extend(expr_result["matches"])
            violations.extend(expr_result["violations"])
            suggestions.extend(expr_result["suggestions"])
        
        elif node_type == "optional":
            # Validate optional step A?
            operand_result = self._validate_ast_node(node["operand"])
            # For optional, we don't add violations if operand fails
            if operand_result["violations"]:
                matches.append(f"Optional step not found (allowed): {node['operand'].get('value', 'unknown')}")
            else:
                matches.extend(operand_result["matches"])
        
        elif node_type in ["choice", "wildcard", "repeat"]:
            # Basic support for other operators
            if "operand" in node:
                operand_result = self._validate_ast_node(node["operand"])
                matches.extend(operand_result["matches"])
                violations.extend(operand_result["violations"])
            elif "left" in node and "right" in node:
                left_result = self._validate_ast_node(node["left"])
                right_result = self._validate_ast_node(node["right"])
                matches.extend(left_result["matches"])
                matches.extend(right_result["matches"])
                violations.extend(left_result["violations"])
                violations.extend(right_result["violations"])
        
        return {
            "matches": matches,
            "violations": violations, 
            "suggestions": suggestions
        }
    
    def _get_spans_from_ast_node(self, node: Dict[str, Any]) -> List[TimelineEvent]:
        """Extract matching spans from AST node for timing validation."""
        if node.get("type") == "identifier":
            return self._find_matching_spans(node["value"])
        elif node.get("type") == "group":
            return self._get_spans_from_ast_node(node["expression"])
        elif node.get("type") in ["sequence", "parallel", "choice"]:
            # For compound nodes, combine spans from both sides
            left_spans = self._get_spans_from_ast_node(node["left"])
            right_spans = self._get_spans_from_ast_node(node["right"])
            return left_spans + right_spans
        elif node.get("type") in ["optional", "wildcard", "repeat"]:
            return self._get_spans_from_ast_node(node["operand"])
        else:
            return []
    
    def _validate_legacy_pattern(self, parsed_pattern: Dict[str, Any]) -> Dict[str, Any]:
        """Legacy validation for backwards compatibility."""
        steps = parsed_pattern["steps"]
        operators = parsed_pattern["operators"]
        
        matches = []
        violations = []
        suggestions = []
        
        if not steps:
            violations.append("Empty pattern - no steps defined")
            return {"matches": matches, "violations": violations, "suggestions": suggestions}
        
        # Find spans matching each step
        step_matches = {}
        for step in steps:
            matching_spans = self._find_matching_spans(step)
            step_matches[step] = matching_spans
            
            if not matching_spans:
                violations.append(f"No spans found matching step: '{step}'")
                suggestions.append(f"Check if step name '{step}' exists in trace. Available span names: {list(set(e.name for e in self.timeline_events))[:5]}")
        
        # Validate sequence and timing
        if len(steps) > 1 and operators:
            sequence_result = self._validate_sequence(steps, operators, step_matches)
            matches.extend(sequence_result["matches"])
            violations.extend(sequence_result["violations"])
            suggestions.extend(sequence_result["suggestions"])
        
        return {
            "matches": matches,
            "violations": violations,
            "suggestions": suggestions,
            "step_matches": step_matches
        }
    
    def _find_matching_spans(self, step_pattern: str) -> List[TimelineEvent]:
        """Find spans that match a step pattern."""
        matching_spans = []
        
        # Support different matching strategies
        for event in self.timeline_events:
            if self._matches_step(event, step_pattern):
                matching_spans.append(event)
                
        return matching_spans
    
    def _matches_step(self, event: TimelineEvent, step_pattern: str) -> bool:
        """Check if an event matches a step pattern (basic span matching)."""
        
        # === EXACT MATCHES ===
        # Exact span name match
        if event.name == step_pattern:
            return True
            
        # Exact span type match
        if event.span_type == step_pattern:
            return True
        
        # === WILDCARD PATTERN MATCHING ===
        if "*" in step_pattern or "?" in step_pattern:
            pattern_regex = step_pattern.replace("*", ".*").replace("?", ".?")
            
            # Test against span name
            if re.match(pattern_regex, event.name, re.IGNORECASE):
                return True
                
            # Test against span type
            if re.match(pattern_regex, event.span_type, re.IGNORECASE):
                return True
        
        # === PARTIAL MATCHING (CONTAINS) ===
        step_lower = step_pattern.lower()
        
        # Partial span name match
        if step_lower in event.name.lower():
            return True
            
        return False
    
    def _validate_sequence(self, steps: List[str], operators: List[str], 
                          step_matches: Dict[str, List[TimelineEvent]]) -> Dict[str, Any]:
        """Validate sequence relationships between steps."""
        matches = []
        violations = []
        suggestions = []
        
        for i in range(len(steps) - 1):
            current_step = steps[i]
            next_step = steps[i + 1]
            
            if i < len(operators):
                operator = operators[i]
                
                current_spans = step_matches.get(current_step, [])
                next_spans = step_matches.get(next_step, [])
                
                if operator == "->":
                    # Validate sequence: current should end before next starts
                    sequence_valid = self._validate_sequential_order(current_spans, next_spans)
                    if sequence_valid["valid"]:
                        matches.append(f"Sequential order validated: {current_step} -> {next_step}")
                    else:
                        violations.extend(sequence_valid["violations"])
                        
                elif operator == "||":
                    # Validate parallel: current and next should overlap in time
                    parallel_valid = self._validate_parallel_execution(current_spans, next_spans)
                    if parallel_valid["valid"]:
                        matches.append(f"Parallel execution validated: {current_step} || {next_step}")
                    else:
                        violations.extend(parallel_valid["violations"])
        
        return {"matches": matches, "violations": violations, "suggestions": suggestions}
    
    def _validate_sequential_order(self, first_spans: List[TimelineEvent], 
                                 second_spans: List[TimelineEvent]) -> Dict[str, Any]:
        """Validate that first spans complete before second spans start."""
        if not first_spans or not second_spans:
            return {"valid": False, "violations": ["Missing spans for sequence validation"]}
            
        violations = []
        
        # Check that at least one first span ends before any second span starts
        first_end_times = [s.end_time for s in first_spans if s.end_time]
        second_start_times = [s.start_time for s in second_spans if s.start_time]
        
        if not first_end_times or not second_start_times:
            violations.append("Missing timing information for sequence validation")
            return {"valid": False, "violations": violations}
            
        earliest_first_end = min(first_end_times)
        latest_second_start = max(second_start_times)
        
        if latest_second_start < earliest_first_end:
            violations.append("Sequence violation: second step started before first step completed")
            
        return {"valid": len(violations) == 0, "violations": violations}
    
    def _validate_parallel_execution(self, first_spans: List[TimelineEvent], 
                                   second_spans: List[TimelineEvent]) -> Dict[str, Any]:
        """Validate that spans execute in parallel (overlapping time periods)."""
        if not first_spans or not second_spans:
            return {"valid": False, "violations": ["Missing spans for parallel validation"]}
            
        violations = []
        
        # Check for time overlap between any first and second spans
        has_overlap = False
        
        for first in first_spans:
            for second in second_spans:
                if (first.start_time and first.end_time and 
                    second.start_time and second.end_time):
                    
                    # Check for time overlap
                    if (first.start_time < second.end_time and 
                        second.start_time < first.end_time):
                        has_overlap = True
                        break
            if has_overlap:
                break
                
        if not has_overlap:
            violations.append("Parallel execution violation: no time overlap found between spans")
            
        return {"valid": len(violations) == 0, "violations": violations}
    
    def suggest_patterns(self) -> List[FlowPattern]:
        """Analyze trace and suggest common flow patterns."""
        if not self.timeline_events:
            return []
            
        patterns = []
        
        # Analyze root events for workflow patterns (events without parents)
        root_events = [e for e in self.timeline_events if e.parent_id is None]
        for root in root_events:
            pattern = self._analyze_event_pattern(root)
            if pattern:
                patterns.append(pattern)
                
        # Suggest common patterns based on span types
        span_types = [e.span_type for e in self.timeline_events]
        type_sequence = " -> ".join(dict.fromkeys(span_types))  # Remove duplicates, preserve order
        
        if len(set(span_types)) > 1:
            patterns.append(FlowPattern(
                name="inferred_type_sequence",
                pattern=type_sequence,
                description=f"Inferred sequence based on span types: {type_sequence}"
            ))
        
        return patterns
    
    def _analyze_event_pattern(self, event: TimelineEvent, level: int = 0) -> Optional[FlowPattern]:
        """Analyze an event and its children to infer patterns."""
        if level > 3:  # Prevent deep recursion
            return None
            
        if not event.children:
            return None
            
        # Simple pattern: sequence of child operations
        child_names = [child.name for child in event.children]
        if len(child_names) > 1:
            pattern_str = " -> ".join(child_names)
            return FlowPattern(
                name=f"{event.name}_workflow",
                pattern=pattern_str,
                description=f"Sequential workflow pattern for {event.name}"
            )
        
        return None
    
    def generate_flow_report(self, patterns: List[Union[str, FlowPattern]]) -> str:
        """Generate a comprehensive flow validation report."""
        lines = []
        lines.append("=== Flow Validation Report ===")
        lines.append("")
        
        if not patterns:
            suggested = self.suggest_patterns()
            lines.append("No patterns provided. Suggested patterns:")
            lines.append("")
            for pattern in suggested[:3]:  # Show top 3
                lines.append(f"Pattern: {pattern.name}")
                lines.append(f"  {pattern.pattern}")
                lines.append(f"  {pattern.description}")
                lines.append("")
            return "\n".join(lines)
        
        all_valid = True
        
        for i, pattern in enumerate(patterns, 1):
            result = self.validate_pattern(pattern)
            
            lines.append(f"Pattern {i}: {result['pattern_name']}")
            lines.append(f"  Expression: {result['pattern']}")
            lines.append(f"  Valid: {'✓' if result['valid'] else '✗'}")
            
            if not result['valid']:
                all_valid = False
                
            if result['matches']:
                lines.append("  Matches:")
                for match in result['matches']:
                    lines.append(f"    ✓ {match}")
                    
            if result['violations']:
                lines.append("  Violations:")
                for violation in result['violations']:
                    lines.append(f"    ✗ {violation}")
                    
            if result['suggestions']:
                lines.append("  Suggestions:")
                for suggestion in result['suggestions']:
                    lines.append(f"    💡 {suggestion}")
                    
            lines.append("")
        
        lines.append("=== Summary ===")
        lines.append(f"Overall validation: {'✓ PASSED' if all_valid else '✗ FAILED'}")
        lines.append(f"Patterns validated: {len(patterns)}")
        
        return "\n".join(lines)