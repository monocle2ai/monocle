"""
Unit tests for Bedrock botocore helper functions.
"""
import pytest
import json
from io import BytesIO
from unittest.mock import Mock, MagicMock
from monocle_apptrace.instrumentation.metamodel.botocore._helper import (
    extract_finish_reason,
    map_finish_reason_to_finish_type
)

class TestBedrockHelperFunctions:
    """Test cases for Bedrock helper functions."""
    
    def test_extract_finish_reason_from_converse_api(self):
        """Test extracting finish reason from Bedrock Converse API response."""
        # Test direct stopReason in result
        arguments = {
            "result": {
                "stopReason": "end_turn"
            },
            "exception": None
        }
        
        result = extract_finish_reason(arguments)
        assert result == "end_turn"
    
    def test_extract_finish_reason_from_output_message(self):
        """Test extracting finish reason from output message structure."""
        arguments = {
            "result": {
                "output": {
                    "message": {
                        "stopReason": "max_tokens"
                    }
                }
            },
            "exception": None
        }
        
        result = extract_finish_reason(arguments)
        assert result == "max_tokens"
    
    def test_extract_finish_reason_from_completion_reason(self):
        """Test extracting finish reason from completionReason field."""
        arguments = {
            "result": {
                "completionReason": "stop_sequence"
            },
            "exception": None
        }
        
        result = extract_finish_reason(arguments)
        assert result == "stop_sequence"
    
    def test_extract_finish_reason_from_nested_result(self):
        """Test extracting finish reason from nested result structure."""
        arguments = {
            "result": {
                "result": {
                    "stopReason": "tool_use"
                }
            },
            "exception": None
        }
        
        result = extract_finish_reason(arguments)
        assert result == "tool_use"
    
    def test_extract_finish_reason_from_response_metadata(self):
        """Test extracting finish reason from ResponseMetadata."""
        arguments = {
            "result": {
                "ResponseMetadata": {
                    "stopReason": "content_filter"
                }
            },
            "exception": None
        }
        
        result = extract_finish_reason(arguments)
        assert result == "content_filter"
    
    def test_extract_finish_reason_from_body_stream(self):
        """Test extracting finish reason from Body stream."""
        # Mock raw stream with JSON data
        response_data = {"stopReason": "end_turn", "answer": "Hello!"}
        response_bytes = json.dumps(response_data).encode('utf-8')
        
        mock_raw_stream = Mock()
        mock_raw_stream.data = response_bytes
        
        mock_body = Mock()
        mock_body._raw_stream = mock_raw_stream
        
        arguments = {
            "result": {
                "Body": mock_body
            },
            "exception": None
        }
        
        result = extract_finish_reason(arguments)
        assert result == "end_turn"
    
    def test_extract_finish_reason_from_streaming_response(self):
        """Test extracting finish reason from streaming response."""
        arguments = {
            "result": {
                "type": "stream",
                "stopReason": "max_tokens"
            },
            "exception": None
        }
        
        result = extract_finish_reason(arguments)
        assert result == "max_tokens"
    
    def test_extract_finish_reason_with_exception(self):
        """Test extracting finish reason when exception is present."""
        arguments = {
            "result": None,
            "exception": Exception("Test error")
        }
        
        result = extract_finish_reason(arguments)
        assert result == "error"
    
    def test_extract_finish_reason_success_default(self):
        """Test default success finish reason when no specific reason found."""
        # Mock get_status_code to return 'success'
        import monocle_apptrace.instrumentation.metamodel.botocore._helper as helper
        original_get_status_code = helper.get_status_code
        helper.get_status_code = lambda x: 'success'
        
        try:
            arguments = {
                "result": {},
                "exception": None
            }
            
            result = extract_finish_reason(arguments)
            assert result == "end_turn"
        finally:
            helper.get_status_code = original_get_status_code
    
    def test_extract_finish_reason_error_default(self):
        """Test default error finish reason when status indicates error."""
        # Mock get_status_code to return 'error'
        import monocle_apptrace.instrumentation.metamodel.botocore._helper as helper
        original_get_status_code = helper.get_status_code
        helper.get_status_code = lambda x: 'error'
        
        try:
            arguments = {
                "result": {},
                "exception": None
            }
            
            result = extract_finish_reason(arguments)
            assert result == "error"
        finally:
            helper.get_status_code = original_get_status_code
    
    def test_extract_finish_reason_none_result(self):
        """Test extracting finish reason when result is None."""
        arguments = {
            "result": None,
            "exception": None
        }
        
        result = extract_finish_reason(arguments)
        assert result is None
    
    def test_extract_finish_reason_malformed_json(self):
        """Test extracting finish reason with malformed JSON in Body."""
        # Mock raw stream with malformed JSON
        response_bytes = b"invalid json {"
        
        mock_raw_stream = Mock()
        mock_raw_stream.data = response_bytes
        
        mock_body = Mock()
        mock_body._raw_stream = mock_raw_stream
        
        arguments = {
            "result": {
                "Body": mock_body
            },
            "exception": None
        }
        
        # Should not raise exception, should return None or default
        result = extract_finish_reason(arguments)
        assert result is None or result == "end_turn"
    
    def test_map_finish_reason_to_finish_type_success(self):
        """Test mapping finish reason to finish type for success cases."""
        test_cases = [
            ("end_turn", "success"),
            ("stop", "success"),
            ("completed", "success"),
            ("tool_use", "tool_call")
        ]
        
        for reason, expected in test_cases:
            result = map_finish_reason_to_finish_type(reason)
            assert result == expected, f"Expected {expected} for {reason}, got {result}"
    
    def test_map_finish_reason_to_finish_type_truncated(self):
        """Test mapping finish reason to finish type for truncated cases."""
        test_cases = [
            ("max_tokens", "truncated"),
            ("length", "truncated"),
            ("token_limit", "truncated")
        ]
        
        for reason, expected in test_cases:
            result = map_finish_reason_to_finish_type(reason)
            assert result == expected, f"Expected {expected} for {reason}, got {result}"
    
    def test_map_finish_reason_to_finish_type_content_filter(self):
        """Test mapping finish reason to finish type for content filter cases."""
        test_cases = [
            ("content_filter", "content_filter"),
            ("safety", "content_filter"),
            ("guardrails", "content_filter"),
            ("blocked", "content_filter")
        ]
        
        for reason, expected in test_cases:
            result = map_finish_reason_to_finish_type(reason)
            assert result == expected, f"Expected {expected} for {reason}, got {result}"
    
    def test_map_finish_reason_to_finish_type_error(self):
        """Test mapping finish reason to finish type for error cases."""
        test_cases = [
            ("error", "error"),
            ("failed", "error"),
            ("exception", "error"),
            ("timeout", "error"),
            ("throttled", "error")
        ]
        
        for reason, expected in test_cases:
            result = map_finish_reason_to_finish_type(reason)
            assert result == expected, f"Expected {expected} for {reason}, got {result}"
    
    def test_map_finish_reason_to_finish_type_none(self):
        """Test mapping None finish reason."""
        result = map_finish_reason_to_finish_type(None)
        assert result is None

if __name__ == "__main__":
    pytest.main([__file__])
