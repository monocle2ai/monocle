from typing import Any, List, Dict, Optional
import re

from llama_index.core.llms import (
    CompletionResponse,
    CompletionResponseGen,
    CustomLLM,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback

class TestChatCompletion:
    
    def __init__(self, usage):
        self.usage = usage 

class TestCompletionUsage:
    
    def __init__(self, completion_tokens, prompt_tokens, total_tokens):
        self.completion_tokens = completion_tokens
        self.prompt_tokens = prompt_tokens 
        self.total_tokens = total_tokens

class OurLLM(CustomLLM):
    context_window: int = 3900
    num_output: int = 256
    model_name: str = "custom"
    dummy_response: str = "My response"

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        return CompletionResponse(
            text=self.dummy_response,
            raw= {
                "usage": TestCompletionUsage(
                    completion_tokens=1,
                    prompt_tokens = 2,
                    total_tokens=3
                )
            })

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponseGen:
        response = ""
        for token in self.dummy_response:
            response += token
            yield CompletionResponse(text=response, delta=token)

def get_span_events_by_name(span, event_name: str) -> List:
    """Get all events of a specific name from a span."""
    return [event for event in span.events if event.name == event_name]

def get_span_event_by_name(span, event_name: str, required: bool = True) -> Optional[Any]:
    """Get the first event of a specific name from a span."""
    events = get_span_events_by_name(span, event_name)
    if not events:
        if required:
            raise AssertionError(f"Expected to find event with name '{event_name}' in span")
        return None
    return events[0]

def verify_span_attributes(span, expected_attributes: Dict[str, Any], optional_attributes: List[str] = None):
    """Verify that a span has the expected attributes."""
    span_attributes = span.attributes
    optional_attributes = optional_attributes or []
    
    for key, expected_value in expected_attributes.items():
        if key not in span_attributes:
            raise AssertionError(f"Expected attribute '{key}' not found in span")
        
        actual_value = span_attributes[key]
        if expected_value is not None and actual_value != expected_value:
            raise AssertionError(f"Expected attribute '{key}' to be '{expected_value}', but got '{actual_value}'")
    
    for key in optional_attributes:
        # Just verify the key exists, don't check value
        if key not in span_attributes:
            raise AssertionError(f"Expected optional attribute '{key}' not found in span")

def verify_inference_span(span, entity_type: str, provider_name: str = None, inference_endpoint: str = None,
                         model_name: str = None, model_type: str = None, check_metadata: bool = True,
                         check_input_output: bool = True):
    """Verify a span is a valid inference span with expected attributes."""
    span_attributes = span.attributes
    
    # Check basic inference attributes
    if "span.type" not in span_attributes or span_attributes["span.type"] not in ["inference", "inference.framework"]:
        raise AssertionError(f"Expected span.type to be 'inference' or 'inference.framework', got {span_attributes.get('span.type')}")
    
    if span_attributes.get("entity.1.type") != entity_type:
        raise AssertionError(f"Expected entity.1.type to be '{entity_type}', got {span_attributes.get('entity.1.type')}")
    
    if provider_name and span_attributes.get("entity.1.provider_name") != provider_name:
        raise AssertionError(f"Expected entity.1.provider_name to be '{provider_name}', got {span_attributes.get('entity.1.provider_name')}")
    
    if inference_endpoint and span_attributes.get("entity.1.inference_endpoint") != inference_endpoint:
        raise AssertionError(f"Expected entity.1.inference_endpoint to be '{inference_endpoint}', got {span_attributes.get('entity.1.inference_endpoint')}")
    
    if model_name and span_attributes.get("entity.2.name") != model_name:
        raise AssertionError(f"Expected entity.2.name to be '{model_name}', got {span_attributes.get('entity.2.name')}")
    
    if model_type and span_attributes.get("entity.2.type") != model_type:
        raise AssertionError(f"Expected entity.2.type to be '{model_type}', got {span_attributes.get('entity.2.type')}")
    
    # Check events
    if check_input_output:
        data_input_event = get_span_event_by_name(span, "data.input")
        data_output_event = get_span_event_by_name(span, "data.output")
        
        if "input" not in data_input_event.attributes:
            raise AssertionError("data.input event should have 'input' attribute")
        
        if "response" not in data_output_event.attributes:
            raise AssertionError("data.output event should have 'response' attribute")
    
    if check_metadata:
        metadata_event = get_span_event_by_name(span, "metadata", required=False)
        if metadata_event:
            required_metadata = ["completion_tokens", "prompt_tokens", "total_tokens"]
            for attr in required_metadata:
                if attr not in metadata_event.attributes:
                    raise AssertionError(f"metadata event should have '{attr}' attribute")

def verify_embedding_span(span, model_name: str):
    span_attributes = span.attributes

    # Must be embedding
    assert span_attributes["span.type"] == "embedding"

    # Entity details
    assert span_attributes["entity.1.name"] == model_name
    assert span_attributes["entity.1.type"] == f"model.embedding.{model_name}"

    # Input/output events
    input_event = next((e for e in span.events if e.name == "data.input"), None)
    output_event = next((e for e in span.events if e.name == "data.output"), None)

    assert input_event is not None
    assert output_event is not None

def find_spans_by_type(spans, span_type: str) -> List:
    """Find all spans of a specific type."""
    return [span for span in spans if span.attributes.get("span.type") == span_type]

def find_span_by_type(spans, span_type: str, required: bool = True) -> Optional[Any]:
    """Find the first span of a specific type."""
    matching_spans = find_spans_by_type(spans, span_type)
    if not matching_spans:
        if required:
            raise AssertionError(f"Expected to find span with type '{span_type}'")
        return None
    return matching_spans[0]

def validate_inference_span_events(span, expected_event_count: int = 3, 
                                 input_patterns: List[str] = None,
                                 output_pattern: str = None,
                                 metadata_requirements: Dict[str, Any] = None):
    """
    Validate inference span events structure and content using regex patterns.
    
    Args:
        span: The span to validate
        expected_event_count: Expected number of events (default 3: input, output, metadata)
        input_patterns: List of regex patterns to validate input tuple elements (assumes input is tuple)
        output_pattern: Regex pattern to validate output string (assumes output is string)
        metadata_requirements: Dict of metadata field requirements:
            - key: field name
            - value: expected type, regex pattern, or callable validator
    """
    events = span.events
    assert len(events) == expected_event_count, f"Expected exactly {expected_event_count} events in the inference span"
    assert events[0].name == "data.input", "First event should be data.input"
    assert events[1].name == "data.output", "Second event should be data.output"
    if expected_event_count >= 3:
        assert events[2].name == "metadata", "Third event should be metadata"

    # Validate input event format
    if input_patterns:
        input_event = events[0].attributes.get("input")
        assert input_event is not None, "Input event should not be None"
        assert isinstance(input_event, tuple), "Input event should be a tuple"
        assert len(input_event) == len(input_patterns), f"Input event should contain exactly {len(input_patterns)} elements"
        
        for i, pattern in enumerate(input_patterns):
            if pattern:  # Skip validation if pattern is None
                assert re.match(pattern, input_event[i]), f"Input element {i} '{input_event[i]}' does not match pattern '{pattern}'"
    
    # Validate output event format
    if output_pattern:
        output_event = events[1].attributes.get("response")
        assert output_event is not None, "Output event should not be None"
        assert isinstance(output_event, str), "Output event should be a string"
        assert re.match(output_pattern, output_event), f"Output event '{output_event}' does not match pattern '{output_pattern}'"
    
    # Validate metadata requirements
    if metadata_requirements and expected_event_count >= 3:
        metadata_event = events[2].attributes
        for field, requirement in metadata_requirements.items():
            assert field in metadata_event, f"Metadata should contain field '{field}'"
            
            value = metadata_event[field]
            if isinstance(requirement, type):
                # Type validation
                assert isinstance(value, requirement), f"Metadata field '{field}' should be of type {requirement.__name__}"
            elif isinstance(requirement, str):
                # Regex pattern validation
                assert re.match(requirement, str(value)), f"Metadata field '{field}' value '{value}' does not match pattern '{requirement}'"
            elif callable(requirement):
                # Custom validator function
                assert requirement(value), f"Metadata field '{field}' value '{value}' failed custom validation"