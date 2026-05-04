"""
Unit tests for SpanFilter functionality.
"""

import pytest
import json
from unittest.mock import Mock, MagicMock
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import SpanContext, TraceFlags
from opentelemetry.sdk.resources import Resource

from monocle_apptrace.exporters.span_filter import SpanFilter, FilteredSpanExporter
from tests.common.span_filter_test_exporter import SpanFilterTestExporter


class TestSpanFilter:
    """Test suite for SpanFilter class."""
    
    @pytest.fixture
    def mock_span(self):
        """Create a mock ReadableSpan for testing."""
        span = Mock(spec=ReadableSpan)
        
        # Mock attributes
        span.attributes = {
            "span.type": "inference",
            "span.subtype": "routing",
            "entity.1.name": "gpt-4",
            "entity.1.type": "inference.openai",
            "entity.2.name": "gpt-4",
            "entity.2.type": "model.llm.gpt-4",
            "scope.customer_id": "cust_123",
            "scope.session_id": "sess_456",
            "monocle_apptrace.version": "1.0.0",
        }
        
        # Mock events as simple dicts (not Mock objects for JSON serializability)
        class EventMock:
            def __init__(self, name, timestamp, attributes):
                self.name = name
                self.timestamp = timestamp
                self.attributes = attributes
        
        span.events = [
            EventMock(
                name="data.input",
                timestamp=1234567890,
                attributes={"input": '[{"role": "user", "content": "Hello"}]'}
            ),
            EventMock(
                name="data.output",
                timestamp=1234567891,
                attributes={"response": '[{"role": "assistant", "content": "Hi there!"}]'}
            ),
            EventMock(
                name="metadata",
                timestamp=1234567892,
                attributes={
                    "completion_tokens": 10,
                    "prompt_tokens": 5,
                    "total_tokens": 15,
                    "finish_reason": "stop"
                }
            ),
        ]
        
        # Mock context
        span.context = SpanContext(
            trace_id=123456789,
            span_id=987654321,
            is_remote=False,
            trace_flags=TraceFlags(0x01)
        )
        
        span.parent = Mock(span_id=111111111)
        span.name = "openai.chat.completions.create"
        span.start_time = 1234567880000000000
        span.end_time = 1234567900000000000
        span.status = Mock(status_code=0)
        span.resource = Resource.create({"service.name": "test-service"})
        
        # Mock to_json
        span.to_json.return_value = json.dumps({
            "name": span.name,
            "context": {
                "trace_id": "000000000000000000000000075bcd15",
                "span_id": "000000003ade68b1",
                "trace_state": "[]"
            },
            "kind": "SpanKind.INTERNAL",
            "parent_id": "0000000006a5f6c7",
            "start_time": span.start_time,
            "end_time": span.end_time,
            "status": {
                "status_code": "UNSET"
            },
            "attributes": span.attributes,
            "events": [
                {
                    "name": event.name,
                    "timestamp": event.timestamp,
                    "attributes": event.attributes
                }
                for event in span.events
            ],
            "resource": {
                "attributes": {
                    "service.name": "test-service"
                }
            }
        })
        
        return span
    
    def test_filter_initialization(self):
        """Test basic filter initialization."""
        config = {
            "span_types_to_include": ["inference"],
            "fields_to_include": {
                "attributes": ["entity.1.name"],
                "events": []
            }
        }
        
        span_filter = SpanFilter(config)
        assert span_filter.span_types_to_include == ["inference"]
        assert span_filter.mode == "include"
    
    def test_filter_initialization_invalid_config(self):
        """Test filter initialization with invalid config."""
        with pytest.raises(ValueError, match="span_types_to_include must be a list"):
            SpanFilter({"span_types_to_include": "not a list"})
        
        with pytest.raises(ValueError, match="fields_to_include must be a dictionary"):
            SpanFilter({
                "span_types_to_include": [],
                "fields_to_include": "not a dict"
            })
    
    def test_should_include_span_exact_match(self, mock_span):
        """Test span inclusion with exact type match."""
        config = {
            "span_types_to_include": ["inference"],
            "fields_to_include": {}
        }
        
        span_filter = SpanFilter(config)
        assert span_filter.should_include_span(mock_span) is True
    
    def test_should_include_span_no_match(self, mock_span):
        """Test span exclusion when type doesn't match."""
        config = {
            "span_types_to_include": ["retrieval"],
            "fields_to_include": {}
        }
        
        span_filter = SpanFilter(config)
        assert span_filter.should_include_span(mock_span) is False
    
    def test_should_include_span_wildcard(self, mock_span):
        """Test span inclusion with wildcard pattern."""
        mock_span.attributes["span.type"] = "inference.framework"
        
        config = {
            "span_types_to_include": ["inference.*"],
            "fields_to_include": {}
        }
        
        span_filter = SpanFilter(config)
        assert span_filter.should_include_span(mock_span) is True
    
    def test_should_include_span_exclude_mode(self, mock_span):
        """Test span exclusion mode."""
        config = {
            "span_types_to_include": ["inference"],
            "mode": "exclude",
            "fields_to_include": {}
        }
        
        span_filter = SpanFilter(config)
        # Should exclude inference types
        assert span_filter.should_include_span(mock_span) is False
        
        # Should include non-inference types
        mock_span.attributes["span.type"] = "retrieval"
        assert span_filter.should_include_span(mock_span) is True
    
    def test_filter_attributes(self, mock_span):
        """Test attribute filtering."""
        config = {
            "span_types_to_include": ["inference"],
            "fields_to_include": {
                "attributes": ["entity.1.name", "entity.2.name", "scope.customer_id"],
                "events": []
            }
        }
        
        span_filter = SpanFilter(config)
        filtered = span_filter.filter(mock_span)
        
        assert filtered is not None
        assert "entity.1.name" in filtered["attributes"]
        assert "entity.2.name" in filtered["attributes"]
        assert "scope.customer_id" in filtered["attributes"]
        
        # These should not be included
        assert "scope.session_id" not in filtered["attributes"]
        assert "monocle_apptrace.version" not in filtered["attributes"]
    
    def test_filter_attributes_wildcard(self, mock_span):
        """Test attribute filtering with wildcard."""
        config = {
            "span_types_to_include": ["inference"],
            "fields_to_include": {
                "attributes": ["scope.*"],
                "events": []
            }
        }
        
        span_filter = SpanFilter(config)
        filtered = span_filter.filter(mock_span)
        
        assert filtered is not None
        assert "scope.customer_id" in filtered["attributes"]
        assert "scope.session_id" in filtered["attributes"]
        
        # Non-scope attributes should not be included
        assert "entity.1.name" not in filtered["attributes"]
    
    def test_filter_events(self, mock_span):
        """Test event filtering."""
        config = {
            "span_types_to_include": ["inference"],
            "fields_to_include": {
                "attributes": [],
                "events": [
                    {"name": "metadata", "attributes": ["completion_tokens", "prompt_tokens"]}
                ]
            }
        }
        
        span_filter = SpanFilter(config)
        filtered = span_filter.filter(mock_span)
        
        assert filtered is not None
        assert len(filtered["events"]) == 1
        assert filtered["events"][0]["name"] == "metadata"
        
        # Only specified attributes should be included
        metadata_attrs = filtered["events"][0]["attributes"]
        assert "completion_tokens" in metadata_attrs
        assert "prompt_tokens" in metadata_attrs
        
        # These should not be included
        assert "total_tokens" not in metadata_attrs
        assert "finish_reason" not in metadata_attrs
    
    def test_filter_multiple_events(self, mock_span):
        """Test filtering multiple events."""
        config = {
            "span_types_to_include": ["inference"],
            "fields_to_include": {
                "attributes": [],
                "events": [
                    {"name": "data.input", "attributes": ["input"]},
                    {"name": "metadata", "attributes": ["completion_tokens"]}
                ]
            }
        }
        
        span_filter = SpanFilter(config)
        filtered = span_filter.filter(mock_span)
        
        assert filtered is not None
        assert len(filtered["events"]) == 2
        
        event_names = [e["name"] for e in filtered["events"]]
        assert "data.input" in event_names
        assert "metadata" in event_names
        assert "data.output" not in event_names
    
    def test_filter_no_matching_span_type(self, mock_span):
        """Test that None is returned for non-matching span types."""
        config = {
            "span_types_to_include": ["retrieval"],
            "fields_to_include": {}
        }
        
        span_filter = SpanFilter(config)
        filtered = span_filter.filter(mock_span)
        
        assert filtered is None
    
    def test_filter_no_field_filtering(self, mock_span):
        """Test that all fields are included when no filtering specified."""
        config = {
            "span_types_to_include": ["inference"],
            "fields_to_include": {}
        }
        
        span_filter = SpanFilter(config)
        filtered = span_filter.filter(mock_span)
        
        assert filtered is not None
        # Should have all original attributes
        assert len(filtered["attributes"]) == len(mock_span.attributes)
        # Should have all original events
        assert len(filtered["events"]) == len(mock_span.events)
    
    def test_filter_multiple_spans(self, mock_span):
        """Test filtering multiple spans."""
        # Create a second span with different type
        mock_span2 = Mock(spec=ReadableSpan)
        mock_span2.attributes = {"span.type": "retrieval"}
        mock_span2.to_json.return_value = json.dumps({
            "name": "retrieval",
            "attributes": mock_span2.attributes,
            "events": []
        })
        
        config = {
            "span_types_to_include": ["inference"],
            "fields_to_include": {}
        }
        
        span_filter = SpanFilter(config)
        filtered_list = span_filter.filter_multiple([mock_span, mock_span2])
        
        # Only inference span should be included
        assert len(filtered_list) == 1
        assert filtered_list[0]["name"] == "openai.chat.completions.create"
    
    def test_matches_pattern(self):
        """Test pattern matching functionality."""
        config = {"span_types_to_include": [], "fields_to_include": {}}
        span_filter = SpanFilter(config)
        
        # Exact match
        assert span_filter._matches_pattern("inference", "inference") is True
        assert span_filter._matches_pattern("inference", "retrieval") is False
        
        # Wildcard match
        assert span_filter._matches_pattern("inference.framework", "inference.*") is True
        assert span_filter._matches_pattern("inference", "inference.*") is True
        assert span_filter._matches_pattern("retrieval", "inference.*") is False
        
        # Full wildcard
        assert span_filter._matches_pattern("anything", "*") is True


class TestFilteredSpanExporter:
    """Test suite for FilteredSpanExporter class."""
    
    @pytest.fixture
    def mock_base_exporter(self):
        """Create a mock base exporter."""
        exporter = Mock()
        exporter.export.return_value = Mock(SUCCESS=0)
        exporter.force_flush.return_value = True
        exporter.shutdown.return_value = None
        return exporter
    
    @pytest.fixture
    def mock_span(self):
        """Create a mock span."""
        span = Mock(spec=ReadableSpan)
        span.attributes = {"span.type": "inference"}
        span.name = "test.span"
        span.context = Mock(trace_id=123, span_id=456)
        span.parent = Mock(span_id=111)
        span.start_time = 1000000000
        span.end_time = 2000000000
        span.status = Mock(status_code=0)
        
        # Mock to_json to return proper JSON string
        span.to_json.return_value = json.dumps({
            "name": span.name,
            "context": {"trace_id": "123", "span_id": "456"},
            "kind": "INTERNAL",
            "parent_id": "111",
            "start_time": span.start_time,
            "end_time": span.end_time,
            "status": {"status_code": "UNSET"},
            "attributes": span.attributes,
            "events": [],
            "resource": {"attributes": {}}
        })
        return span
    
    def test_filtered_exporter_delegates_export(self, mock_base_exporter, mock_span):
        """Test that filtered exporter delegates to base exporter."""
        config = {
            "span_types_to_include": ["inference"],
            "fields_to_include": {}
        }
        
        span_filter = SpanFilter(config)
        filtered_exporter = FilteredSpanExporter(mock_base_exporter, span_filter)
        
        filtered_exporter.export([mock_span])
        
        # Should delegate to base exporter
        mock_base_exporter.export.assert_called_once()
        assert len(mock_base_exporter.export.call_args[0][0]) == 1
    
    def test_filtered_exporter_filters_spans(self, mock_base_exporter, mock_span):
        """Test that non-matching spans are filtered out."""
        config = {
            "span_types_to_include": ["retrieval"],  # Different from span type
            "fields_to_include": {}
        }
        
        span_filter = SpanFilter(config)
        filtered_exporter = FilteredSpanExporter(mock_base_exporter, span_filter)
        
        filtered_exporter.export([mock_span])
        
        # Should not call base exporter since no spans match
        mock_base_exporter.export.assert_not_called()
    
    def test_filtered_exporter_force_flush(self, mock_base_exporter):
        """Test that force_flush is delegated."""
        config = {"span_types_to_include": [], "fields_to_include": {}}
        span_filter = SpanFilter(config)
        filtered_exporter = FilteredSpanExporter(mock_base_exporter, span_filter)
        
        result = filtered_exporter.force_flush(10000)
        
        assert result is True
        mock_base_exporter.force_flush.assert_called_once_with(10000)
    
    def test_filtered_exporter_shutdown(self, mock_base_exporter):
        """Test that shutdown is delegated."""
        config = {"span_types_to_include": [], "fields_to_include": {}}
        span_filter = SpanFilter(config)
        filtered_exporter = FilteredSpanExporter(mock_base_exporter, span_filter)
        
        filtered_exporter.shutdown()
        
        mock_base_exporter.shutdown.assert_called_once()


class TestSpanFilterIntegration:
    """Integration tests with example configurations."""
    
    def test_example1_configuration(self):
        """Test the example configuration from requirements."""
        # Example 1 configuration
        config = {
            "span_types_to_include": ["inference", "inference.framework"],
            "fields_to_include": {
                "attributes": ["entity.1.name", "entity.2.name", "scope.customer_id"],
                "events": [
                    {"name": "metadata", "attributes": ["completion_tokens", "prompt_tokens"]}
                ]
            }
        }
        
        span_filter = SpanFilter(config)
        
        # Verify configuration is valid
        assert span_filter.span_types_to_include == ["inference", "inference.framework"]
        assert len(span_filter.attribute_patterns) == 3
        assert len(span_filter.event_configs) == 1
    
    def test_cost_tracking_use_case(self):
        """Test a cost tracking use case (tokens only)."""
        config = {
            "span_types_to_include": ["inference*"],  # All inference types
            "fields_to_include": {
                "attributes": ["entity.1.name", "entity.2.name", "scope.*"],
                "events": [
                    {
                        "name": "metadata",
                        "attributes": ["completion_tokens", "prompt_tokens", "total_tokens"]
                    }
                ]
            }
        }
        
        span_filter = SpanFilter(config)
        
        # Should match inference and inference.framework
        mock_span = Mock(spec=ReadableSpan)
        mock_span.attributes = {"span.type": "inference.framework"}
        assert span_filter.should_include_span(mock_span) is True
    
    def test_error_tracking_use_case(self):
        """Test an error tracking use case."""
        config = {
            "span_types_to_include": ["*"],  # All span types
            "fields_to_include": {
                "attributes": ["span.type", "entity.1.name", "scope.*"],
                "events": [
                    {"name": "data.output", "attributes": ["error_code", "response"]}
                ]
            }
        }
        
        span_filter = SpanFilter(config)
        
        # Should include all span types
        mock_span = Mock(spec=ReadableSpan)
        mock_span.attributes = {"span.type": "agentic.tool.invocation"}
        assert span_filter.should_include_span(mock_span) is True


# ---------------------------------------------------------------------------
# Tests that use SpanFilterTestExporter
# ---------------------------------------------------------------------------

def _make_span(span_type: str, name: str, extra_attrs: dict = None, events: list = None) -> Mock:
    """Build a minimal mock ReadableSpan suitable for SpanFilter tests."""
    span = Mock(spec=ReadableSpan)
    attrs = {"span.type": span_type, **(extra_attrs or {})}
    span.attributes = attrs
    span.name = name
    span.context = SpanContext(
        trace_id=111111111, span_id=222222222,
        is_remote=False, trace_flags=TraceFlags(0x01),
    )
    span.parent = None
    span.start_time = 1_000_000_000
    span.end_time = 2_000_000_000
    span.status = Mock(status_code=0)
    span.resource = Resource.create({"service.name": "test-svc"})

    raw_events = events or []
    span.to_json.return_value = json.dumps({
        "name": name,
        "context": {"trace_id": "0000000006a5f6c7", "span_id": "000000000de0b6b3", "trace_state": "[]"},
        "kind": "SpanKind.INTERNAL",
        "parent_id": None,
        "start_time": span.start_time,
        "end_time": span.end_time,
        "status": {"status_code": "UNSET"},
        "attributes": attrs,
        "events": raw_events,
        "resource": {"attributes": {"service.name": "test-svc"}},
    })
    return span


class TestSpanFilterTestExporter:
    """Tests for SpanFilterTestExporter used as the base_exporter in FilteredSpanExporter."""

    def _make_inference_span(self):
        return _make_span(
            span_type="inference",
            name="openai.chat",
            extra_attrs={
                "entity.1.type": "inference.openai",
                "entity.1.name": "gpt-4",
                "scope.customer_id": "cust_001",
                "scope.session_id": "sess_abc",
                "monocle_apptrace.version": "1.0.0",
            },
            events=[
                {"name": "data.input", "timestamp": 1, "attributes": {"input": "hello"}},
                {"name": "metadata", "timestamp": 2, "attributes": {
                    "completion_tokens": 10,
                    "prompt_tokens": 5,
                    "total_tokens": 15,
                    "finish_reason": "stop",
                }},
            ],
        )

    def _make_retrieval_span(self):
        return _make_span(
            span_type="retrieval",
            name="chroma.query",
            extra_attrs={"entity.1.name": "vector-db", "scope.customer_id": "cust_001"},
        )

    # --- standalone usage (no FilteredSpanExporter wrapper) ---

    def test_standalone_captures_matching_spans(self):
        test_exporter = SpanFilterTestExporter(
            span_filter=SpanFilter({
                "span_types_to_include": ["inference"],
                "fields_to_include": {
                    "attributes": ["entity.1.type", "scope.*"],
                    "events": [{"name": "metadata"}],
                },
            })
        )

        spans = [self._make_inference_span(), self._make_retrieval_span()]
        test_exporter.export(spans)

        test_exporter.assert_exported_count(1)
        test_exporter.assert_attribute_present(0, "entity.1.type")
        test_exporter.assert_attribute_present(0, "scope.customer_id")
        test_exporter.assert_attribute_absent(0, "monocle_apptrace.version")
        test_exporter.assert_event_present(0, "metadata")
        test_exporter.assert_event_absent(0, "data.input")

    def test_standalone_excludes_all_when_no_match(self):
        test_exporter = SpanFilterTestExporter(
            span_filter=SpanFilter({
                "span_types_to_include": ["workflow"],
                "fields_to_include": {},
            })
        )
        test_exporter.export([self._make_inference_span(), self._make_retrieval_span()])
        test_exporter.assert_exported_count(0)

    # --- usage via FilteredSpanExporter ---

    def test_with_filtered_exporter_captures_filtered_content(self):
        test_exporter = SpanFilterTestExporter()
        filtered_exporter = FilteredSpanExporter(
            base_exporter=test_exporter,
            span_filter=SpanFilter({
                "span_types_to_include": ["inference"],
                "fields_to_include": {
                    "attributes": ["entity.1.name", "scope.*"],
                    "events": [{"name": "metadata", "attributes": ["completion_tokens"]}],
                },
            }),
        )

        filtered_exporter.export([self._make_inference_span(), self._make_retrieval_span()])

        test_exporter.assert_exported_count(1)
        test_exporter.assert_attribute_present(0, "entity.1.name")
        test_exporter.assert_attribute_present(0, "scope.customer_id")
        # attribute not in filter config — must be absent
        test_exporter.assert_attribute_absent(0, "entity.1.type")

        # only completion_tokens requested from metadata
        event_attrs = test_exporter.get_event_attributes(0, "metadata")
        assert "completion_tokens" in event_attrs
        assert "prompt_tokens" not in event_attrs

    def test_with_filtered_exporter_exclude_mode(self):
        test_exporter = SpanFilterTestExporter()
        filtered_exporter = FilteredSpanExporter(
            base_exporter=test_exporter,
            span_filter=SpanFilter({
                "span_types_to_include": ["retrieval"],
                "mode": "exclude",
                "fields_to_include": {},
            }),
        )

        filtered_exporter.export([self._make_inference_span(), self._make_retrieval_span()])

        # retrieval span excluded → only inference span exported
        test_exporter.assert_exported_count(1)
        spans = test_exporter.find_spans_by_type("inference")
        assert len(spans) == 1

    def test_reset_clears_captured_spans(self):
        test_exporter = SpanFilterTestExporter(
            span_filter=SpanFilter({"span_types_to_include": ["inference"], "fields_to_include": {}})
        )
        test_exporter.export([self._make_inference_span()])
        assert test_exporter.export_count == 1

        test_exporter.reset()
        assert test_exporter.export_count == 0

    def test_find_spans_by_name(self):
        test_exporter = SpanFilterTestExporter(
            span_filter=SpanFilter({"span_types_to_include": ["inference"], "fields_to_include": {}})
        )
        test_exporter.export([self._make_inference_span()])

        matches = test_exporter.find_spans_by_name("openai.chat")
        assert len(matches) == 1

        no_matches = test_exporter.find_spans_by_name("nonexistent")
        assert len(no_matches) == 0

    def test_assertion_helpers_raise_on_failure(self):
        test_exporter = SpanFilterTestExporter(
            span_filter=SpanFilter({
                "span_types_to_include": ["inference"],
                "fields_to_include": {"attributes": ["entity.1.name"], "events": []},
            })
        )
        test_exporter.export([self._make_inference_span()])

        with pytest.raises(AssertionError):
            test_exporter.assert_exported_count(99)

        with pytest.raises(AssertionError):
            test_exporter.assert_attribute_present(0, "nonexistent_key")

        with pytest.raises(AssertionError):
            test_exporter.assert_attribute_absent(0, "entity.1.name")  # it IS present
