"""
Agent-specific assertion plugins for agent workflow and execution validation.

This module contains plugins that provide assertions specifically for validating
agent behavior, tool usage, and multi-agent workflows.
"""
from typing import TYPE_CHECKING

from monocle_tfwk.assertions.flow_validator import FlowValidator
from monocle_tfwk.assertions.plugin_registry import TraceAssertionsPlugin, plugin
from monocle_tfwk.schema import EntityType, ESpanAttribute, MonocleSpanType

if TYPE_CHECKING:
    from monocle_tfwk.assertions.trace_assertions import TraceAssertions


# Module-level constants for agent flow validation
TARGET_TYPES = [
    MonocleSpanType.AGENTIC_DELEGATION, 
    MonocleSpanType.AGENTIC_TOOL_INVOCATION, 
    MonocleSpanType.AGENTIC_MCP_INVOCATION
]

# Common patterns for agent detection (constants to reduce duplication)
USER_PATTERNS = ["user", "human", "input", "request"]


def _has_entity_names(*entities) -> bool:
    """Helper function to check if all entities have names."""
    return all(entity and 'name' in entity for entity in entities)


def _has_first_two_entity_names(entities: list) -> bool:
    """Helper function to check if first two entities have names."""
    return (len(entities) >= 2 and 
            'name' in entities[0] and 
            'name' in entities[1])


def _extract_entities_from_attributes(attributes: dict) -> list:
    """
    Extract entities array from span attributes.
    
    Returns: [{'name': '', 'type': '', 'from_agent': '', 'to_agent': '', ...}, ...]
    """
    entities = []
    
    # Find all entity numbers by looking for entity.X. prefixes
    entity_numbers = set()
    for key in attributes.keys():
        if key.startswith('entity.'):
            parts = key.split('.')
            if len(parts) >= 2 and parts[1].isdigit():
                entity_numbers.add(int(parts[1]))
    
    # Build entities array
    for i in sorted(entity_numbers):
        entity = {}
        # Collect all attributes for this entity
        for key, value in attributes.items():
            if key.startswith(f'entity.{i}.'):
                attr_name = key.replace(f'entity.{i}.', '')
                entity[attr_name] = value
        
        if entity:  # Only add if entity has attributes
            entities.append(entity)
    
    return entities


class AgentFlowValidator(FlowValidator):
    """
    Enhanced FlowValidator with agent and tool pattern matching capabilities.
    
    Extends the base FlowValidator to support matching by:
    - Agent names (from entities array and legacy agent.name)
    - Tool names (from entities array and legacy tool.name)  
    - Agent types (from entities array and legacy agent.type)
    - Tool types (from entities array and legacy tool.type)
    
    Note: This class includes fallback mechanisms for backward compatibility:
    - Legacy field support (agent.name, tool.name) for older traces
    - Multiple matching strategies for robustness
    - Flexible direction handling for tool invocations
    """
    
    def _matches_step(self, event, step_pattern: str) -> bool:
        """
        Enhanced step matching that includes agent and tool patterns.
        
        First tries the base FlowValidator matching (span type and name),
        then tries agent/tool specific patterns if no match found.
        """
        # Try base matching first (span type and span name)
        if super()._matches_step(event, step_pattern):
            return True
            
        # If base matching fails, try agent/tool specific matching
        return self._matches_agent_or_tool_step(event, step_pattern)
    
    def _matches_agent_or_tool_step(self, event, step_pattern: str) -> bool:
        """
        Enhanced matching logic for agent and tool patterns.
        
        Checks agent names, tool names, agent types, and tool types
        in addition to standard span type and name matching.
        """
        # Extract entities once and reuse for all matching operations
        entities = self._extract_entities_from_attributes(event.attributes)
        step_pattern_lower = step_pattern.lower()
        
        # Single iteration through entities for all matching types
        for entity in entities:
            entity_type = entity.get('type', '')
            
            # Check all entity field values for name matching
            for field_value in entity.values():
                if isinstance(field_value, str) and field_value.lower() == step_pattern_lower:
                    return True
            
            # Check entity type matching (for agent type patterns)
            if entity_type and entity_type.lower() == step_pattern_lower:
                return True
        
        return False

    def _validate_tuple_agent_flow(self, tuple_pattern: list) -> dict:
        """
        Validate tuple-based agent flow pattern: [(participant1, participant2, span_type), ...]
        
        This is the ONLY supported format for agent flow validation.
        Each tuple represents an interaction between two participants with a specific span type.
        
        Args:
            tuple_pattern: List of tuples in format (participant1, participant2, span_type)
                          e.g., [("U", "TC", "request"), ("TC", "FA", "delegation")]
        """
        result = {
            "valid": False,
            "matches": [],
            "violations": [],
            "suggestions": []
        }
        
        if not tuple_pattern:
            result["violations"].append("Empty tuple pattern provided")
            return result
        
        # Validate that all items are tuples with 3 elements
        for i, item in enumerate(tuple_pattern):
            if not isinstance(item, tuple) or len(item) != 3:
                result["violations"].append(
                    f"Pattern item {i} must be a tuple with 3 elements (participant1, participant2, span_type), got: {item}"
                )
                return result
        
        # Try to match each tuple pattern against timeline events
        matched_tuples = []
        unmatched_tuples = []
        
        for pattern_tuple in tuple_pattern:
            participant1, participant2, expected_span_type = pattern_tuple
            
            # Look for events that match this interaction pattern
            found_match = False
            for t_event in self.timeline_events:
                if self._matches_tuple_interaction(t_event, participant1, participant2, expected_span_type):
                    matched_tuples.append({
                        "pattern": pattern_tuple,
                        "event": t_event,
                        "span_type": t_event.attributes.get(ESpanAttribute.SPAN_TYPE, "unknown"),
                        "participants": self._extract_participants_from_event(t_event)
                    })
                    found_match = True
                    break
            
            if not found_match:
                unmatched_tuples.append(pattern_tuple)
        
        # Set validation result
        if unmatched_tuples:
            result["violations"].extend([
                f"No matching event found for interaction: {tuple_pattern}" for tuple_pattern in unmatched_tuples
            ])
            result["suggestions"].append(f"Available interactions: {self._get_available_interactions()}")
        else:
            result["valid"] = True
            
        result["matches"] = matched_tuples
        return result
    
    def _matches_tuple_interaction(self, event, participant1: str, participant2: str, expected_span_type: str) -> bool:
        """
        Check if an event matches a tuple interaction pattern.
        
        Args:
            event: Timeline event to check
            participant1: First participant (caller/initiator)
            participant2: Second participant (callee/target)
            expected_span_type: Expected span type for this interaction
        """
        # More flexible span type matching
        event_span_type = event.attributes.get(ESpanAttribute.SPAN_TYPE, "").lower()
        event_name = event.name.lower() if event.name else ""
        
        # Flexible span type matching
        span_type_matches = (
            expected_span_type.lower() in event_span_type or
            expected_span_type.lower() in event_name 
        )
        
        if not span_type_matches:
            return False
        
        # Extract participants from event
        event_participants = self._extract_participants_from_event(event)
        
        # More flexible participant matching (check if either participant is involved)
        participant_matches = (
            any(participant1.lower() in str(p).lower() for p in event_participants) or
            any(participant2.lower() in str(p).lower() for p in event_participants) or
            # Check event name for participant references using helper
            self._check_participant_in_event_name(event_name, participant1) or
            self._check_participant_in_event_name(event_name, participant2)
        )
        
        return participant_matches
    
    def _extract_participants_from_event(self, event) -> list:
        """Extract participant names from a timeline event."""
        participants = []
        
        # Extract participants using entities array approach
        entities = self._extract_entities_from_attributes(event.attributes)
        
        for entity in entities:
            # Add all string values from entities
            for field_value in entity.values():
                if isinstance(field_value, str) and field_value and field_value not in participants:
                    participants.append(field_value)
        
        # Add event name as potential participant reference
        self._add_event_name_as_participant(event.name, participants)
            
        return participants
    
    def _add_event_name_as_participant(self, event_name: str, participants: list) -> None:
        """Helper method to add event name as participant if not already present."""
        if event_name and event_name not in participants:
            participants.append(event_name)
    
    def _check_event_name_for_pattern(self, event_name: str, patterns: list) -> bool:
        """Helper method to check if event name contains any of the given patterns."""
        if not event_name:
            return False
        event_name_lower = event_name.lower()
        return any(pattern in event_name_lower for pattern in patterns)
    
    def _check_participant_in_event_name(self, event_name: str, participant: str) -> bool:
        """Helper method to check if participant is referenced in event name."""
        if not event_name or not participant:
            return False
        return participant.lower() in event_name.lower()
    
    def _get_available_interactions(self) -> list:
        """Get list of available interactions from timeline events for debugging."""
        interactions = []
        for t_event in self.timeline_events[:10]:  # Show more events for better debugging
            participants = self._extract_participants_from_event(t_event)
            span_type = t_event.attributes.get(ESpanAttribute.SPAN_TYPE, "unknown")
            event_name = t_event.name if t_event.name else "unknown"
            
            # Include both multi-participant and single-participant events for debugging
            if participants:
                if len(participants) >= 2:
                    interactions.append((participants[0], participants[1], span_type))
                else:
                    # Single participant events might still be relevant
                    interactions.append((participants[0], "?", f"{span_type}:{event_name}"))
                    
        return interactions

    def _validate_flow_definition(self, flow_def: dict) -> dict:
        """
        Validate comprehensive flow definition with participants registry.
        
        Args:
            flow_def: Flow definition dictionary with participants, required, forbidden, interactions
            
        Returns:
            dict: Validation result with valid flag, matches, violations, and suggestions
        """
        result = {
            "valid": False,
            "matches": [],
            "violations": [],
            "suggestions": []
        }
        
        participants = flow_def.get("participants", [])
        required_participants = flow_def.get("required", [])
        forbidden_interactions = flow_def.get("forbidden", [])
        interactions = flow_def.get("interactions", [])
        
        # Build participant registry from definition
        participant_registry = {}
        for participant in participants:
            if isinstance(participant, tuple) and len(participant) >= 4:
                alias, name, ptype, description = participant[:4]
                participant_registry[alias] = {
                    "name": name, 
                    "type": ptype, 
                    "description": description,
                    "found": False  # Track if we found this participant in traces
                }
        
        # Check required participants
        for required_alias in required_participants:
            if required_alias not in participant_registry:
                result["violations"].append(f"Required participant '{required_alias}' not defined in participants list")
                return result
        
        # Validate each interaction
        matched_interactions = []
        unmatched_interactions = []
        
        for interaction in interactions:
            if not isinstance(interaction, tuple) or len(interaction) != 3:
                result["violations"].append(f"Invalid interaction format: {interaction}. Expected (from, to, type)")
                continue
                
            from_alias, to_alias, interaction_type = interaction
            
            # Find matching events for this interaction
            found_match = False
            for t_event in self.timeline_events:
                if self._matches_interaction_with_participant_awareness(
                    t_event, from_alias, to_alias, interaction_type, participant_registry
                ):
                    matched_interactions.append({
                        "interaction": interaction,
                        "event": t_event,
                        "span_type": t_event.attributes.get(ESpanAttribute.SPAN_TYPE, "unknown"),
                        "relationship": self._analyze_span_relationship(t_event)
                    })
                    
                    # Mark participants as found
                    if from_alias in participant_registry:
                        participant_registry[from_alias]["found"] = True
                    if to_alias in participant_registry:
                        participant_registry[to_alias]["found"] = True
                        
                    found_match = True
                    break
            
            if not found_match:
                unmatched_interactions.append(interaction)
        
        # Check forbidden interactions
        forbidden_violations = []
        for forbidden in forbidden_interactions:
            if isinstance(forbidden, tuple) and len(forbidden) == 2:
                from_alias, to_alias = forbidden
                # Check if any events match this forbidden pattern
                for t_event in self.timeline_events:
                    if self._matches_forbidden_interaction(t_event, from_alias, to_alias, participant_registry):
                        forbidden_violations.append({
                            "forbidden": forbidden,
                            "violating_event": t_event
                        })
        
        # Check required participants presence
        missing_required = []
        for required_alias in required_participants:
            if required_alias in participant_registry and not participant_registry[required_alias]["found"]:
                missing_required.append(required_alias)
        
        # Compile results
        if unmatched_interactions:
            result["violations"].extend([
                f"No matching event found for interaction: {interaction}" 
                for interaction in unmatched_interactions
            ])
        
        if forbidden_violations:
            result["violations"].extend([
                f"Forbidden interaction detected: {violation['forbidden']} in event {violation['violating_event'].name}"
                for violation in forbidden_violations
            ])
        
        if missing_required:
            result["violations"].extend([
                f"Required participant '{alias}' not found in traces" 
                for alias in missing_required
            ])
        
        if not result["violations"]:
            result["valid"] = True
        
        result["matches"] = matched_interactions
        result["suggestions"] = self._get_flow_suggestions(participant_registry)
        
        return result
    
    def _matches_interaction_with_participant_awareness(
        self, event, from_alias: str, to_alias: str, interaction_type: str, participant_registry: dict
    ) -> bool:
        """
        Enhanced interaction matching with participant registry awareness.
        
        Considers both sibling and parent-child span relationships.
        """
        # First check if the interaction type matches
        if not self._matches_interaction_type(event, interaction_type):
            return False
        
        # Special handling for 'request' type interactions from 'User' or 'Start'
        if interaction_type.lower() == "request" and from_alias.lower() in ["user", "u", "start"]:
            # For request interactions, check if any child span has agent invocation
            return self._has_child_agent_invocation(event)
        
        # Special handling for agentic.delegation - check from_agent and to_agent fields
        event_span_type = event.attributes.get(ESpanAttribute.SPAN_TYPE, "").lower()
        if MonocleSpanType.AGENTIC_DELEGATION in event_span_type or interaction_type.lower() == "delegation":
            return self._matches_delegation_flow(event, from_alias, to_alias, participant_registry)
        
        # Special handling for agentic.tool.invocation - direction is always agent -> tool
        if MonocleSpanType.AGENTIC_TOOL_INVOCATION in event_span_type or interaction_type.lower() == "invocation":
            return self._matches_tool_invocation_flow(event, from_alias, to_alias, participant_registry)
        
        # Get participant details from registry
        from_participant = participant_registry.get(from_alias, {})
        to_participant = participant_registry.get(to_alias, {})
        
        # Extract participants from event using enhanced matching
        event_participants = self._extract_enhanced_participants_from_event(event, participant_registry)
        
        # Check if both participants are represented in this event
        from_found = self._participant_matches_event(event, from_participant, from_alias, event_participants)
        to_found = self._participant_matches_event(event, to_participant, to_alias, event_participants)
        
        return from_found and to_found
    
    def _matches_tool_invocation_flow(self, event, from_alias: str, to_alias: str, participant_registry: dict) -> bool:
        """
        Special handling for agentic.tool.invocation where flow is always agent -> tool.
        
        For tool invocation spans, we normalize the direction so that:
        - Agent is always the "from" participant 
        - Tool is always the "to" participant
        
        This handles cases where entity.1 or entity.2 might contain the agent/tool in either order.
        """
        # Get participant details from registry
        from_participant = participant_registry.get(from_alias, {})
        to_participant = participant_registry.get(to_alias, {})
        
        # For agentic.tool.invocation, check both directions since the trace might have either:
        # Case 1: Expected pattern - agent -> tool (from_alias=agent, to_alias=tool)
        # Case 2: Actual trace pattern - tool -> agent (but we want to match it as agent -> tool)
        
        from_is_agent = from_participant.get("type") == EntityType.AGENT
        to_is_tool = to_participant.get("type") == EntityType.TOOL
        from_is_tool = from_participant.get("type") == EntityType.TOOL  
        to_is_agent = to_participant.get("type") == EntityType.AGENT
        
        # Extract entities once using the centralized method
        entities = self._extract_entities_from_attributes(event.attributes)
        
        # Case 1: Expected agent -> tool flow
        if from_is_agent and to_is_tool:
            agent_name_expected = from_participant.get("name", "")
            tool_name_expected = to_participant.get("name", "")
            
            # Check if both agent and tool are found in the entities array
            agent_found = self._name_found_in_entities(agent_name_expected, from_alias, entities)
            tool_found = self._name_found_in_entities(tool_name_expected, to_alias, entities)
            
            return agent_found and tool_found
            
        # Case 2: Reverse pattern - if user expects tool -> agent, still match if both are present
        elif from_is_tool and to_is_agent:
            agent_name_expected = to_participant.get("name", "")
            tool_name_expected = from_participant.get("name", "")
            
            # Check if both tool and agent are found in the entities array
            agent_found = self._name_found_in_entities(agent_name_expected, to_alias, entities)
            tool_found = self._name_found_in_entities(tool_name_expected, from_alias, entities)
            
            return agent_found and tool_found
        
        # If not the expected agent->tool flow, fall back to regular matching
        return False
    
    def _name_found_in_entities(self, expected_name: str, alias: str, entities: list) -> bool:
        """Check if expected name or alias is found in any entity from the entities array."""
        expected_name_lower = expected_name.lower()
        alias_lower = alias.lower()
        
        for entity in entities:
            # Check entity name field
            entity_name = entity.get('name', '').lower()
            if entity_name and (expected_name_lower in entity_name or alias_lower in entity_name):
                return True
            
            # Check entity type field (for type-based matching)
            entity_type = entity.get('type', '').lower()
            if entity_type and (expected_name_lower in entity_type or alias_lower in entity_type):
                return True
                
            # Check other entity fields like from_agent, to_agent, description, etc.
            for field_value in entity.values():
                if isinstance(field_value, str):
                    field_lower = field_value.lower()
                    if expected_name_lower in field_lower or alias_lower in field_lower:
                        return True
        
        return False
    
    def _matches_delegation_flow(self, event, from_alias: str, to_alias: str, participant_registry: dict) -> bool:
        """
        Special handling for agentic.delegation using entities array approach.
        
        For delegation spans, we extract from_agent and to_agent from the entities array.
        """
        # Extract entities using the centralized method
        entities = self._extract_entities_from_attributes(event.attributes)
        
        # Find from_agent and to_agent from entities
        from_agent = ""
        to_agent = ""
        
        for entity in entities:
            if 'from_agent' in entity and entity['from_agent']:
                from_agent = entity['from_agent']
            if 'to_agent' in entity and entity['to_agent']:
                to_agent = entity['to_agent']
        
        # Get participant details from registry
        from_participant = participant_registry.get(from_alias, {})
        to_participant = participant_registry.get(to_alias, {})
        
        # Check if the from_agent matches the expected from participant
        from_name = from_participant.get("name", "").lower()
        from_matches = (
            (from_alias.lower() == from_agent.lower()) or
            (from_name == from_agent.lower()) or
            (from_alias.lower() in from_agent.lower()) or
            (from_agent.lower() in from_name if from_name else False)
        )
        
        # Check if the to_agent matches the expected to participant  
        to_name = to_participant.get("name", "").lower()
        to_matches = (
            (to_alias.lower() == to_agent.lower()) or
            (to_name == to_agent.lower()) or
            (to_alias.lower() in to_agent.lower()) or
            (to_agent.lower() in to_name if to_name else False)
        )
        
        return from_matches and to_matches
    
    def _has_child_agent_invocation(self, event) -> bool:
        """
        Check if this event or any child span has agentic invocation span types.
        
        For 'request' type interactions, we look for evidence of agent invocation
        by checking for MonocleSpanType.AGENTIC_INVOCATION specifically.
        """
        # Check current event for agentic invocation span type
        event_span_type = event.attributes.get(ESpanAttribute.SPAN_TYPE, "")
        
        # Look for specific agentic invocation span type
        if event_span_type == MonocleSpanType.AGENTIC_INVOCATION:
            return True
        
        # Check for agent entities using proper entity type
        entities = self._extract_entities_from_attributes(event.attributes)
        for entity in entities:
            # Check entity type for proper agent classification
            entity_type = entity.get('type', '')
            if entity_type and EntityType.AGENT in entity_type.lower():
                return True
        
        
        # If we have timeline events available, check child spans
        if hasattr(self, 'timeline_events'):
            current_span_id = event.span_id if hasattr(event, 'span_id') else None
            current_trace_id = event.trace_id if hasattr(event, 'trace_id') else None
            
            if current_span_id and current_trace_id:
                for t_child_event in self.timeline_events:
                    # Check if this is a child span
                    child_parent_id = getattr(t_child_event, 'parent_span_id', None) if hasattr(t_child_event, 'parent_span_id') else None
                    child_trace_id = getattr(t_child_event, 'trace_id', None) if hasattr(t_child_event, 'trace_id') else None
                    
                    if (child_parent_id == current_span_id and child_trace_id == current_trace_id):
                        # Check child for agentic invocation span type
                        child_span_type = t_child_event.attributes.get(ESpanAttribute.SPAN_TYPE, "")
                        
                        if child_span_type == MonocleSpanType.AGENTIC_INVOCATION:
                            return True
                            
                        # Check child for agent entities using proper entity type
                        child_entities = self._extract_entities_from_attributes(t_child_event.attributes)
                        for entity in child_entities:
                            entity_type = entity.get('type', '')  
                            if entity_type and EntityType.AGENT in entity_type.lower():
                                return True
        
        return False
    
    def _matches_interaction_type(self, event, interaction_type: MonocleSpanType) -> bool:
        """Check if event matches the expected interaction type."""
        event_span_type = event.attributes.get(ESpanAttribute.SPAN_TYPE, "").lower()
        event_name = event.name.lower() if event.name else ""
        
        # Since MonocleSpanType is a StrEnum, we can directly use its string value
        interaction_type_lower = interaction_type.value.lower()
        
        # Direct matching - exact span type or partial match
        return interaction_type_lower in event_span_type or interaction_type_lower in event_name
    
    def _extract_enhanced_participants_from_event(self, event, participant_registry: dict) -> dict:
        """Extract participants from event with enhanced participant registry awareness."""
        participants = {}
        
        # Extract participants using entities array approach
        entities = self._extract_entities_from_attributes(event.attributes)
        
        for entity in entities:
            # Check entity name
            entity_name = entity.get('name', '')
            if entity_name:
                # Try to match against participant registry
                for alias, participant_info in participant_registry.items():
                    if (entity_name.lower() == participant_info["name"].lower() or 
                        entity_name.lower() == alias.lower()):
                        participants[alias] = entity_name
                        break
            
            # Check other entity fields like from_agent, to_agent
            for field_name, field_value in entity.items():
                if isinstance(field_value, str) and field_value:
                    # Try to match field values against participant registry
                    for alias, participant_info in participant_registry.items():
                        if (field_value.lower() == participant_info["name"].lower() or 
                            field_value.lower() == alias.lower()):
                            participants[alias] = field_value
                            break
        
        # Check event name for participant references using helper
        if event.name:
            for alias, participant_info in participant_registry.items():
                if (self._check_participant_in_event_name(event.name, alias) or 
                    self._check_participant_in_event_name(event.name, participant_info["name"])):
                    participants[alias] = event.name
        
        return participants
    
    def _participant_matches_event(self, event, participant_info: dict, alias: str, event_participants: dict) -> bool:
        """Check if a participant is represented in an event."""
        # Direct alias match
        if alias in event_participants:
            return True
        
        # Type-based matching
        return self._matches_participant_by_type(event, participant_info)
    
    def _matches_participant_by_type(self, event, participant_info: dict) -> bool:
        """Helper method to match participant by type."""
        participant_type = participant_info.get("type", "")
        participant_name = participant_info.get("name", "").lower()
        event_name_lower = event.name.lower() if event.name else ""
        
        if participant_type == EntityType.TOOL:
            return "tool" in event_name_lower and participant_name in event_name_lower
        elif participant_type == EntityType.AGENT:
            return "agent" in event_name_lower and participant_name in event_name_lower
        elif participant_type == "actor":
            # Check for user/human-related events
            return any(term in event_name_lower for term in USER_PATTERNS)
        
        return False
    
    def _matches_forbidden_interaction(self, event, from_alias: str, to_alias: str, participant_registry: dict) -> bool:
        """Check if event represents a forbidden interaction between participants."""
        from_participant = participant_registry.get(from_alias, {})
        to_participant = participant_registry.get(to_alias, {})
        
        event_participants = self._extract_enhanced_participants_from_event(event, participant_registry)
        
        from_found = self._participant_matches_event(event, from_participant, from_alias, event_participants)
        to_found = self._participant_matches_event(event, to_participant, to_alias, event_participants)
        
        return from_found and to_found
    
    def _analyze_span_relationship(self, event) -> dict:
        """
        Analyze span relationships (sibling vs child) for enhanced validation.
        
        Returns information about whether this is a parent span, child span, or sibling.
        """
        relationship_info = {
            "type": "unknown",
            "has_parent": event.parent_id is not None,
            "parent_id": event.parent_id,
            "is_root": event.parent_id is None,
        }
        
        # Determine relationship type based on parent-child structure
        if event.parent_id is None:
            relationship_info["type"] = "root"
        else:
            # Check if there are other events with the same parent (siblings)
            siblings = [t_e for t_e in self.timeline_events if t_e.parent_id == event.parent_id and t_e.span_id != event.span_id]
            relationship_info["sibling_count"] = len(siblings)
            relationship_info["type"] = "child" if len(siblings) == 0 else "sibling"
        
        # Check if this event has children
        children = [t_e for t_e in self.timeline_events if t_e.parent_id == event.span_id]
        relationship_info["has_children"] = len(children) > 0
        relationship_info["child_count"] = len(children)
        
        return relationship_info
    
    def _get_flow_suggestions(self, participant_registry: dict) -> list:
        """Generate helpful suggestions based on available events and participant registry."""
        suggestions = []
        
        # Show available participants found in traces
        found_participants = [alias for alias, info in participant_registry.items() if info["found"]]
        if found_participants:
            suggestions.append(f"Found participants in traces: {found_participants}")
        
        # Show available interactions from timeline events (limited for readability)
        available_interactions = self._get_available_interactions()[:5]  # Limit to 5 examples
        if available_interactions:
            suggestions.append(f"Sample available interactions: {available_interactions}")
        
        return suggestions
    
    def _extract_entities_from_attributes(self, attributes: dict) -> list:
        """
        Extract entities array from span attributes.
        
        Returns: [{'name': '', 'type': '', 'from_agent': '', 'to_agent': '', ...}, ...]
        """
        entities = []
        
        # Find all entity numbers by looking for entity.X. prefixes
        entity_numbers = set()
        for key in attributes.keys():
            if key.startswith('entity.'):
                parts = key.split('.')
                if len(parts) >= 2 and parts[1].isdigit():
                    entity_numbers.add(int(parts[1]))
        
        # Build entities array
        for i in sorted(entity_numbers):
            entity = {}
            # Collect all attributes for this entity
            for key, value in attributes.items():
                if key.startswith(f'entity.{i}.'):
                    attr_name = key.replace(f'entity.{i}.', '')
                    entity[attr_name] = value
            
            if entity:  # Only add if entity has attributes
                entities.append(entity)
        
        return entities


@plugin
class AgentAssertionsPlugin(TraceAssertionsPlugin):
    """Plugin providing agent-specific assertion methods."""
    
    @classmethod
    def get_plugin_name(cls) -> str:
        return EntityType.AGENT
    
    def assert_agent_called(self, agent_name: str) -> 'TraceAssertions':
        """Assert that an agent with the given name was called."""
        agent_spans = self.get_agents_by_name(agent_name)
        assert len(agent_spans) > 0, f"Agent '{agent_name}' was not called. Available agents: {self.get_agent_names()}"
        return self
        
    def get_called_tools(self) -> list[str]:
        """
        Get a list of all tool names that were called in the current traces.
        
        Returns:
            List of tool names that were invoked during the traced execution
        """
        tool_names = []
        
        for span in self._current_spans:
            # Check legacy attribute first
            legacy_tool = span.attributes.get("tool.name")
            if legacy_tool and legacy_tool not in tool_names:
                tool_names.append(legacy_tool)
                continue
                
            # Check entities array for tool names
            entities = self._extract_entities_from_attributes(span.attributes)
            for entity in entities:
                if 'tool' in entity.get('type', '').lower():
                    tool_name = entity.get('name', '')
                    if tool_name and tool_name not in tool_names:
                        tool_names.append(tool_name)
        
        return tool_names

    def called_tool(self, tool_name: str) -> 'TraceAssertions':
        """Assert that a tool with the given name was called."""
        matching_spans = []
        
        for span in self._current_spans:     
            # Check entities array for tool name
            entities = self._extract_entities_from_attributes(span.attributes)
            for entity in entities:
                if (entity.get('name') == tool_name and 
                    'tool' in entity.get('type', '').lower()):
                    matching_spans.append(span)
                    break
        
        assert matching_spans, f"No tool named '{tool_name}' found in traces"
        self._current_spans = matching_spans
        return self

    def assert_agent_type(self, agent_type: str) -> 'TraceAssertions':
        """Assert that traces contain a specific agent type using JMESPath."""
        found = self.query_engine.has_agent_type(agent_type)
        assert found, f"Agent type '{agent_type}' not found in traces"
        return self
    
    
    
    def assert_agent_flow(self, expected_pattern: dict) -> 'TraceAssertions':
        """
        Assert agent flow using JMESPath-based branch collection and validation.
        
        Algorithm:
        1. Collect branches under agentic.invocation that end with agentic.delegation, agentic.tool.invocation, or agentic.mcp.invocation
        2. For each branch, collapse to tuple (from, to, span.type of leaf) with span_ids list
        3. Sort tuples by start time if they are siblings
        4. For an agentic.request, check these tuples against expected interactions
        
        Args:
            expected_pattern: Flow definition dict
        
        Returns:
            TraceAssertions: Self for method chaining
        """
        # Create a reference to the standalone function for nested functions
        extract_entities = _extract_entities_from_attributes
        
        def _collect_agentic_branches() -> list:
            """
            Collect branches including both interaction and request spans.
            
            Returns branches that include:
            1. agentic.delegation, agentic.tool.invocation, agentic.mcp.invocation (leaf spans)
            2. agentic.request -> agentic.invocation (user request to first agent)
            """
            branches = []
            
            # Find all target span types regardless of whether they're leaf spans
            # All these spans represent important interactions in the agent flow
            type_conditions = [f'attributes."{ESpanAttribute.SPAN_TYPE}" == \'{t}\'' for t in TARGET_TYPES]
            target_spans_query = f"[?{' || '.join(type_conditions)}]"
            target_spans = self.query_engine.query(target_spans_query) or []
            
            # Find agentic.request spans (user requests)
            request_spans_query = f"[?attributes.\"{ESpanAttribute.SPAN_TYPE}\" == '{MonocleSpanType.AGENTIC_REQUEST}']"
            request_spans = self.query_engine.query(request_spans_query) or []
            
            # Find agentic.invocation spans (agent invocations)
            invocation_spans_query = f"[?attributes.\"{ESpanAttribute.SPAN_TYPE}\" == '{MonocleSpanType.AGENTIC_INVOCATION}']"
            invocation_spans = self.query_engine.query(invocation_spans_query) or []
            
            # Debug: Let's see what span types are actually available
            all_span_types_query = f"[].attributes.\"{ESpanAttribute.SPAN_TYPE}\" | [?@ != null] | sort(@)"
            all_span_types = self.query_engine.query(all_span_types_query) or []
            unique_span_types = list(set(all_span_types))
            print(f"DEBUG: Available span types: {unique_span_types}")
            print(f"DEBUG: Found {len(target_spans)} target spans, {len(request_spans)} request spans, {len(invocation_spans)} invocation spans")
            
            # Debug: Show which spans were found
            print("DEBUG: Collected spans found:")
            for i, span in enumerate(target_spans):
                span_type = span.get('attributes', {}).get(ESpanAttribute.SPAN_TYPE, 'UNKNOWN')
                span_id = span.get('span_id', 'NO_ID')
                print(f"  {i+1}. {span_type} (ID: {span_id})")
                if span_type in TARGET_TYPES:
                    entities = extract_entities(span.get('attributes', {}))
                    print(f"     Entities: {entities}")
            
            # Add target spans as branches
            for target_span in target_spans:
                branch = _trace_branch_to_invocation(target_span)
                if branch:
                    branches.append(branch)
            
            # Handle agentic.request -> agentic.invocation pattern
            # For each request span, find the first invocation span under it (or related to it)
            for request in request_spans:
                # Find first agentic.invocation that might be related to this request
                # For now, we'll use the first invocation span found
                if invocation_spans:
                    first_invocation = invocation_spans[0]  # Take first invocation
                    # Create a synthetic branch representing User -> First Agent
                    synthetic_branch = {
                        'leaf': first_invocation,
                        'path': [request, first_invocation],
                        'root_type': 'agentic.request',
                        'span_ids': [request.get('span_id', 'unknown'), first_invocation.get('span_id', 'unknown')],
                        'request_span': request,
                        'invocation_span': first_invocation
                    }
                    branches.append(synthetic_branch)
            
            return branches
        
        def _trace_branch_to_invocation(leaf_span: dict) -> dict:
            """
            Trace a leaf span back to see if it's under an agentic.invocation.
            
            Returns branch info with full path from invocation to leaf.
            """
            # For now, simplified approach - find direct relationships
            # In a full implementation, we'd traverse the parent-child hierarchy
            
            # Check if leaf span has indicators of being part of an agentic flow
            span_type = leaf_span.get('attributes', {}).get(ESpanAttribute.SPAN_TYPE, '')
            
            if any(agentic_type in span_type for agentic_type in TARGET_TYPES):
                return {
                    'leaf': leaf_span,
                    'path': [leaf_span],  # Simplified - in full impl would include full parent chain
                    'root_type': MonocleSpanType.AGENTIC_INVOCATION,  # Assumed for now
                    'span_ids': [leaf_span.get('span_id', 'unknown')]
                }
            
            # Handle agentic.request spans - they don't need to be under invocation
            if MonocleSpanType.AGENTIC_REQUEST in span_type:
                return {
                    'leaf': leaf_span,
                    'path': [leaf_span],  
                    'root_type': MonocleSpanType.AGENTIC_REQUEST,
                    'span_ids': [leaf_span.get('span_id', 'unknown')]
                }
            
            return None
        
        def _collapse_branches_to_tuples(branches: list) -> list:
            """
            Collapse branches to tuples (from, to, span.type) with metadata.
            
            For each branch, extract:
            - from: source entity (from entities array)
            - to: target entity (from entities array)
            - span.type: the leaf span type
            - span_ids: list of span IDs in the branch
            - start_time: for sorting
            """
            tuples = []
            
            for branch in branches:
                leaf = branch['leaf']
                attributes = leaf.get('attributes', {})
                
                # Check if this is a synthetic request -> invocation branch
                if branch.get('root_type') == 'agentic.request' and branch.get('invocation_span'):
                    # Special handling for User -> Agent request pattern
                    invocation_span = branch['invocation_span']
                    invocation_attributes = invocation_span.get('attributes', {})
                    entities = _extract_entities_from_attributes(invocation_attributes)
                    
                    # Get agent name from entities array
                    agent_name = 'Agent'  # default
                    for entity in entities:
                        if entity.get('name'):
                            agent_name = entity['name']
                            break
                    
                    tuple_info = {
                        'tuple': ('U', agent_name, MonocleSpanType.AGENTIC_REQUEST.value),  # User -> Agent request with string value
                        'span_ids': branch['span_ids'],
                        'start_time': leaf.get('start_time', 0),
                        'attributes': attributes,
                        'leaf_span': leaf
                    }
                    tuples.append(tuple_info)
                else:
                    # Regular span processing
                    span_type = attributes.get(ESpanAttribute.SPAN_TYPE, '')
                    from_participant, to_participant = _extract_participants_from_span(attributes, span_type)
                    
                    if from_participant and to_participant:
                        tuple_info = {
                            'tuple': (from_participant, to_participant, span_type),  # Use full span type
                            'span_ids': branch['span_ids'],
                            'start_time': leaf.get('start_time', 0),
                            'attributes': attributes,
                            'leaf_span': leaf
                        }
                        tuples.append(tuple_info)
            
            return tuples
        def _extract_participants_from_span(attributes: dict, span_type: str) -> tuple:
            """
            Extract from and to participants based on span type and attributes.
            
            Returns (from_participant, to_participant)
            """
            entities = extract_entities(attributes)
            
            if MonocleSpanType.AGENTIC_DELEGATION in span_type:
                # Extract from_agent and to_agent from entities
                from_agent = ''
                to_agent = ''
                for entity in entities:
                    if 'from_agent' in entity and entity['from_agent']:
                        from_agent = entity['from_agent']
                    if 'to_agent' in entity and entity['to_agent']:
                        to_agent = entity['to_agent']
                
                return (from_agent, to_agent) if from_agent and to_agent else (None, None)

            elif (MonocleSpanType.AGENTIC_TOOL_INVOCATION in span_type or MonocleSpanType.AGENTIC_MCP_INVOCATION in span_type):
                if len(entities) < 2:
                    return (None, None)
                
                # Find agent and tool entities by type
                agent_entity = None
                tool_entity = None
                
                for entity in entities:
                    entity_type = entity.get('type', '')
                    if EntityType.AGENT in entity_type:
                        agent_entity = entity
                    elif EntityType.TOOL in entity_type:
                        tool_entity = entity
                
                # Return agent -> tool order if both found with names
                if _has_entity_names(agent_entity, tool_entity):
                    return (agent_entity['name'], tool_entity['name'])
                
                # Fallback: use first two entities if they have names
                if _has_first_two_entity_names(entities):
                    return (entities[0]['name'], entities[1]['name'])
                
                return (None, None)
            
            
            return (None, None)
        
        
        def _sort_tuples_by_start_time(tuples: list) -> list:
            """
            Sort tuples by start time, especially important for siblings.
            """
            return sorted(tuples, key=lambda t: t['start_time'])
        
        def _convert_interactions_to_full_names(expected_pattern: dict):
            """Convert all alias-based interactions to use full names from participant registry."""
            # Build registry from participants
            participants_registry = {}
            for participant in expected_pattern.get('participants', []):
                if isinstance(participant, tuple) and len(participant) >= 4:
                    alias, name, ptype, description = participant[:4]
                    participants_registry[alias] = {"name": name, "type": ptype, "description": description}
            
            # Convert interactions to use full names
            converted_interactions = []
            for interaction in expected_pattern.get('interactions', []):
                if len(interaction) == 3:
                    from_alias, to_alias, interaction_type = interaction
                    from_name = participants_registry.get(from_alias, {}).get('name', from_alias)
                    to_name = participants_registry.get(to_alias, {}).get('name', to_alias)
                    # Convert enum to string value if needed
                    type_str = interaction_type.value if hasattr(interaction_type, 'value') else str(interaction_type)
                    converted_interactions.append((from_name, to_name, type_str))
            
            return converted_interactions

        def _validate_interaction_tuples(interaction_tuples: list, expected_pattern):
            """
            Validate collected interaction tuples against expected pattern.
            """
            # Extract just the tuple parts for comparison
            actual_interactions = [t['tuple'] for t in interaction_tuples]
            
            # Convert expected interactions to full names at the start
            expected_interactions = _convert_interactions_to_full_names(expected_pattern)
            
            # Validate each expected interaction
            violations = []
            for expected in expected_interactions:
                if not _find_matching_interaction(expected, actual_interactions):
                    violations.append(f"No matching event found for interaction: {expected}")
            
            if violations:
                available_info = f"Available interactions found in traces: {actual_interactions}"
                error_message = f"Agent flow validation failed: {violations}\n{available_info}"
                raise AssertionError(error_message)
        
        def _find_matching_interaction(expected: tuple, actual_interactions: list) -> bool:
            """Find if expected interaction matches any actual interaction."""
            expected_from, expected_to, expected_type = expected
            
            for actual in actual_interactions:
                actual_from, actual_to, actual_type = actual
                
                # Convert actual type to string for comparison
                actual_type_str = actual_type.value if hasattr(actual_type, 'value') else str(actual_type)
                
                # Check type match (case-insensitive)
                if not _interaction_types_match(expected_type, actual_type_str):
                    continue
                
                # Check participant matches (case-insensitive, partial matching)
                from_matches = (
                    expected_from.lower() == actual_from.lower() or
                    expected_from.lower() in actual_from.lower() or
                    actual_from.lower() in expected_from.lower()
                )
                
                to_matches = (
                    expected_to.lower() == actual_to.lower() or
                    expected_to.lower() in actual_to.lower() or
                    actual_to.lower() in expected_to.lower()
                )
                
                if from_matches and to_matches:
                    return True
            
            return False


        

        
        def _interaction_types_match(expected_type, actual_type) -> bool:
            """Check if interaction types match with exact enum/string comparison."""
            # Convert to string values if enum objects
            expected_str = expected_type.value if hasattr(expected_type, 'value') else str(expected_type)
            actual_str = actual_type.value if hasattr(actual_type, 'value') else str(actual_type)
            
            expected_lower = expected_str.lower()
            actual_lower = actual_str.lower()
            
            # Direct exact match or partial match
            return expected_lower == actual_lower or expected_lower in actual_lower or actual_lower in expected_lower
        
        # Step 1: Collect branches using JMESPath queries
        branches = _collect_agentic_branches()
        
        # Step 2: Collapse branches to tuples with metadata
        interaction_tuples = _collapse_branches_to_tuples(branches)
        
        # Step 3: Sort by start time for siblings
        sorted_tuples = _sort_tuples_by_start_time(interaction_tuples)
        
        # Step 4: Validate against expected pattern
        _validate_interaction_tuples(sorted_tuples, expected_pattern)
        
        return self