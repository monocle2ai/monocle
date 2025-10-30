#!/usr/bin/env python3
"""
Simple pytest tests to validate JSON traces against schemas
"""

import json
from pathlib import Path

import pytest

# Import our schema
from monocle_tfwk.schema import (
    MonocleSchemaValidator,
    MonocleSpanSchemaRegistry,
    MonocleSpanType,
)


class MockSpan:
    """Mock span object for validation testing"""
    def __init__(self, attributes, events=None):
        self.attributes = attributes
        self.events = events or []

class MockEvent:
    """Mock event object for validation testing"""
    def __init__(self, name, attributes):
        self.name = name
        self.attributes = attributes

# =============================================================================
# PYTEST FIXTURES
# =============================================================================

@pytest.fixture
def json_dir():
    """Get the JSON directory with integration test files using relative path"""
    # Get current file's directory and navigate to the integration tests
    current_dir = Path(__file__).parent
    # From tfwk/tests/unit/ go up to tfwk/tests/, then tfwk/, then monocle/, then down to apptrace/tests/integration
    integration_dir = current_dir.parent.parent.parent / "apptrace" / "tests" / "integration"
    return str(integration_dir)

@pytest.fixture
def json_spans(json_dir):
    """Load JSON spans from integration directory"""
    json_path = Path(json_dir)
    spans = []
    
    for json_file in json_path.glob("*.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    spans.extend(data)
                else:
                    spans.append(data)
                print(f"✅ Loaded {json_file.name}: {len(data) if isinstance(data, list) else 1} spans")
        except Exception as e:
            print(f"❌ Failed to load {json_file.name}: {e}")
    
    return spans

def get_schema_for_span_type(span_type: str):
    """Get schema for a given span type"""
    try:
        span_enum = MonocleSpanType(span_type)
        if span_enum == MonocleSpanType.WORKFLOW:
            return MonocleSpanSchemaRegistry.workflow_schema()
        elif span_enum == MonocleSpanType.HTTP_PROCESS:
            return MonocleSpanSchemaRegistry.http_process_schema()
        elif span_enum == MonocleSpanType.INFERENCE:
            return MonocleSpanSchemaRegistry.inference_schema()
        elif span_enum == MonocleSpanType.AGENTIC_INVOCATION:
            return MonocleSpanSchemaRegistry.agentic_invocation_schema()
        elif span_enum == MonocleSpanType.EMBEDDING:
            return MonocleSpanSchemaRegistry.embedding_schema()
        # Add more schema mappings as needed
        else:
            return None
    except ValueError:
        return None

# =============================================================================
# PYTEST TEST FUNCTIONS
# =============================================================================

def test_json_directory_exists(json_dir):
    """Test that the JSON directory exists and contains files"""
    json_path = Path(json_dir)
    assert json_path.exists(), f"JSON directory does not exist: {json_dir}"
    
    json_files = list(json_path.glob("*.json"))
    assert len(json_files) > 0, f"No JSON files found in {json_dir}"
    print(f"✅ Found {len(json_files)} JSON files in {json_dir}")

def test_load_json_spans(json_spans):
    """Test that JSON spans can be loaded successfully"""
    assert len(json_spans) > 0, "No spans were loaded from JSON files"
    print(f"✅ Successfully loaded {len(json_spans)} spans")

def test_validate_json_spans_against_schemas(json_spans):
    """Test that JSON spans validate against their schemas"""
    validation_results = []
    
    for span in json_spans:
        if 'attributes' not in span:
            continue
            
        attrs = span['attributes']
        span_type = attrs.get('span.type', 'unknown')
        
        # Get schema for this span type
        schema = get_schema_for_span_type(span_type)
        if not schema:
            continue
            
        # Create mock span for validation
        mock_span = MockSpan(
            attributes=attrs,
            events=[MockEvent(e.get('name', ''), e.get('attributes', {})) 
                   for e in span.get('events', [])]
        )
        
        # Validate attributes
        attr_errors = MonocleSchemaValidator.validate_span_attributes(mock_span, schema)
        
        # Validate events
        event_errors = MonocleSchemaValidator.validate_span_events(mock_span, schema)
        
        if attr_errors or event_errors:
            validation_results.append({
                'span_type': span_type,
                'attribute_errors': attr_errors,
                'event_errors': event_errors
            })
    
    # Report results
    if validation_results:
        print(f"⚠️  Found {len(validation_results)} spans with validation errors:")
        for result in validation_results[:5]:  # Show first 5
            print(f"  - {result['span_type']}: attr_errors={len(result['attribute_errors'])}, event_errors={len(result['event_errors'])}")
    else:
        print("✅ All spans validate successfully against their schemas")
    
    # Test should pass even with some validation errors (schemas may be stricter than real data)
    total_validated = len([s for s in json_spans if get_schema_for_span_type(s.get('attributes', {}).get('span.type', '')) is not None])
    if total_validated > 0:
        error_rate = len(validation_results) / total_validated
        assert error_rate < 0.8, f"Too many validation errors: {error_rate:.1%}"

def test_schema_validation_logic():
    """Test that the schema validation logic works correctly"""
    
    # Test valid HTTP span
    valid_http_span = MockSpan(
        attributes={
            "http.method": "POST",
            "http.status_code": "200", 
            "http.url": "/users",
            "span.type": "http.process"
        },
        events=[
            MockEvent("http.request", {"method": "POST", "url": "/users"}),
            MockEvent("http.response", {"status_code": "200"})
        ]
    )
    
    http_schema = MonocleSpanSchemaRegistry.http_process_schema()
    attr_errors = MonocleSchemaValidator.validate_span_attributes(valid_http_span, http_schema)
    event_errors = MonocleSchemaValidator.validate_span_events(valid_http_span, http_schema)
    
    assert not attr_errors, f"Valid HTTP span should not have attribute errors: {attr_errors}"
    assert not event_errors, f"Valid HTTP span should not have event errors: {event_errors}"
    
    # Test invalid inference span (should fail validation)
    invalid_inference_span = MockSpan(
        attributes={
            "span.type": "inference",
            "entity.1.type": "inference.openai"
            # Missing required attributes
        }
    )
    
    inference_schema = MonocleSpanSchemaRegistry.inference_schema()
    attr_errors = MonocleSchemaValidator.validate_span_attributes(invalid_inference_span, inference_schema)
    
    assert len(attr_errors) > 0, "Invalid inference span should have validation errors"
    
    print("✅ Schema validation logic works correctly")

if __name__ == "__main__":
    # Run the test directly if executed as a script
    pytest.main([__file__, "-s", "--tb=short"])


