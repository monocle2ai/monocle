"""
Unit tests for Bedrock finish reason mapping functions.
"""
import pytest
from monocle_apptrace.instrumentation.metamodel.finish_types import (
    map_bedrock_finish_reason_to_finish_type,
    BEDROCK_FINISH_REASON_MAPPING,
    FinishType
)

class TestBedrockFinishReasonMapping:
    """Test cases for Bedrock finish reason mapping."""
    
    def test_success_reasons(self):
        """Test that success reasons map to 'success' finish type."""
        success_reasons = [
            "end_turn",
            "stop",
            "stop_sequence",
            "completed",
            "endoftext",
            "COMPLETE",
            "FINISH"
        ]
        
        for reason in success_reasons:
            result = map_bedrock_finish_reason_to_finish_type(reason)
            assert result == FinishType.SUCCESS.value, f"Expected 'success' for {reason}, got {result}"
    
    def test_tool_call_reasons(self):
        """Test that tool call reasons map to 'tool_call' finish type."""
        tool_call_reasons = [
            "tool_use",
            "function_call"
        ]
        
        for reason in tool_call_reasons:
            result = map_bedrock_finish_reason_to_finish_type(reason)
            assert result == FinishType.TOOL_CALL.value, f"Expected 'tool_call' for {reason}, got {result}"
    
    def test_truncated_reasons(self):
        """Test that truncated reasons map to 'truncated' finish type."""
        truncated_reasons = [
            "max_tokens",
            "length",
            "max_length",
            "token_limit",
            "MAX_TOKENS",
            "LENGTH"
        ]
        
        for reason in truncated_reasons:
            result = map_bedrock_finish_reason_to_finish_type(reason)
            assert result == FinishType.TRUNCATED.value, f"Expected 'truncated' for {reason}, got {result}"
    
    def test_content_filter_reasons(self):
        """Test that content filter reasons map to 'content_filter' finish type."""
        content_filter_reasons = [
            "content_filter",
            "content_filtered",
            "safety",
            "guardrails",
            "blocked",
            "CONTENT_FILTERED"
        ]
        
        for reason in content_filter_reasons:
            result = map_bedrock_finish_reason_to_finish_type(reason)
            assert result == FinishType.CONTENT_FILTER.value, f"Expected 'content_filter' for {reason}, got {result}"
    
    def test_error_reasons(self):
        """Test that error reasons map to 'error' finish type."""
        error_reasons = [
            "error",
            "failed",
            "exception",
            "timeout",
            "model_error",
            "service_unavailable",
            "throttled",
            "rate_limit",
            "validation_error",
            "ERROR"
        ]
        
        for reason in error_reasons:
            result = map_bedrock_finish_reason_to_finish_type(reason)
            assert result == FinishType.ERROR.value, f"Expected 'error' for {reason}, got {result}"
    
    def test_case_insensitive_mapping(self):
        """Test that mapping is case insensitive."""
        test_cases = [
            ("END_TURN", FinishType.SUCCESS.value),
            ("end_turn", FinishType.SUCCESS.value),
            ("End_Turn", FinishType.SUCCESS.value),
            ("MAX_TOKENS", FinishType.TRUNCATED.value),
            ("max_tokens", FinishType.TRUNCATED.value),
            ("Max_Tokens", FinishType.TRUNCATED.value),
            ("CONTENT_FILTER", FinishType.CONTENT_FILTER.value),
            ("content_filter", FinishType.CONTENT_FILTER.value),
            ("Content_Filter", FinishType.CONTENT_FILTER.value),
            ("ERROR", FinishType.ERROR.value),
            ("error", FinishType.ERROR.value),
            ("Error", FinishType.ERROR.value)
        ]
        
        for reason, expected in test_cases:
            result = map_bedrock_finish_reason_to_finish_type(reason)
            assert result == expected, f"Expected {expected} for {reason}, got {result}"
    
    def test_pattern_matching(self):
        """Test that unknown reasons are matched by patterns."""
        pattern_test_cases = [
            ("custom_stop_sequence", FinishType.SUCCESS.value),
            ("model_finished_successfully", FinishType.SUCCESS.value),
            ("token_limit_exceeded", FinishType.TRUNCATED.value),
            ("max_length_reached", FinishType.TRUNCATED.value),
            ("safety_filter_activated", FinishType.CONTENT_FILTER.value),
            ("content_blocked_by_guardrails", FinishType.CONTENT_FILTER.value),
            ("network_error_occurred", FinishType.ERROR.value),
            ("service_failure", FinishType.ERROR.value),
            ("request_throttled", FinishType.ERROR.value)
        ]
        
        for reason, expected in pattern_test_cases:
            result = map_bedrock_finish_reason_to_finish_type(reason)
            assert result == expected, f"Expected {expected} for {reason}, got {result}"
    
    def test_none_and_empty_input(self):
        """Test handling of None and empty input."""
        assert map_bedrock_finish_reason_to_finish_type(None) is None
        assert map_bedrock_finish_reason_to_finish_type("") is None
        assert map_bedrock_finish_reason_to_finish_type(" ") is None
    
    def test_unknown_reason_returns_none(self):
        """Test that completely unknown reasons return None."""
        unknown_reasons = [
            "unknown_reason",
            "random_string",
            "xyz123",
            "1234"
        ]
        
        for reason in unknown_reasons:
            result = map_bedrock_finish_reason_to_finish_type(reason)
            assert result is None, f"Expected None for unknown reason {reason}, got {result}"
    
    def test_model_specific_reasons(self):
        """Test model-specific finish reasons."""
        model_specific_cases = [
            # Claude models
            ("end_turn", FinishType.SUCCESS.value),
            ("tool_use", FinishType.TOOL_CALL.value),
            
            # AI21 models
            ("endoftext", FinishType.SUCCESS.value),
            
            # Cohere models
            ("COMPLETE", FinishType.SUCCESS.value),
            ("MAX_TOKENS", FinishType.TRUNCATED.value),
            
            # Titan models
            ("FINISH", FinishType.SUCCESS.value),
            ("LENGTH", FinishType.TRUNCATED.value),
            ("CONTENT_FILTERED", FinishType.CONTENT_FILTER.value)
        ]
        
        for reason, expected in model_specific_cases:
            result = map_bedrock_finish_reason_to_finish_type(reason)
            assert result == expected, f"Expected {expected} for {reason}, got {result}"

if __name__ == "__main__":
    pytest.main([__file__])
