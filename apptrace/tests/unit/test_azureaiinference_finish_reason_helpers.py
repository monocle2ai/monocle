"""
Unit tests for Azure AI Inference helper functions, particularly finish_reason mapping.
"""
import unittest
from types import SimpleNamespace

from monocle_apptrace.instrumentation.metamodel.azureaiinference._helper import (
    extract_finish_reason,
    map_finish_reason_to_finish_type,
)
from monocle_apptrace.instrumentation.metamodel.finish_types import (
    AZURE_AI_INFERENCE_FINISH_REASON_MAPPING,
)


class TestAzureAIInferenceFinishReasonHelpers(unittest.TestCase):
    """Test Azure AI Inference finish reason extraction and mapping functions."""

    def test_finish_reason_mapping_completeness(self):
        """Test that all expected Azure AI Inference finish reasons are mapped."""
        expected_reasons = {
            # Standard completion reasons
            "stop": "success",
            "completed": "success",
            "finished": "success",
            
            # Token limits
            "length": "truncated",
            "max_tokens": "truncated",
            "token_limit": "truncated",
            "max_completion_tokens": "truncated",
            
            # Tool/function calling
            "tool_calls": "tool_call",
            "function_call": "tool_call",
            
            # Content filtering and safety
            "content_filter": "content_filter",
            "content_filtered": "content_filter",
            "safety": "content_filter",
            "responsible_ai_policy": "content_filter",
            
            # Errors
            "error": "error",
            "failed": "error",
            "exception": "error",
            "timeout": "error",
            
            # Azure-specific reasons
            "model_error": "error",
            "service_unavailable": "error",
            "rate_limit": "error",
        }
        
        # Check that our mapping contains all expected keys
        for key, value in expected_reasons.items():
            self.assertIn(key, AZURE_AI_INFERENCE_FINISH_REASON_MAPPING)
            self.assertEqual(AZURE_AI_INFERENCE_FINISH_REASON_MAPPING[key], value)

    def test_map_finish_reason_to_finish_type_success_cases(self):
        """Test mapping of success-type finish reasons."""
        success_reasons = ["stop", "completed", "finished"]
        tool_call_reasons = ["tool_calls", "function_call"]
        for reason in success_reasons:
            with self.subTest(reason=reason):
                self.assertEqual(map_finish_reason_to_finish_type(reason), "success")
        for reason in tool_call_reasons:
            with self.subTest(reason=reason):
                self.assertEqual(map_finish_reason_to_finish_type(reason), "tool_call")

    def test_map_finish_reason_to_finish_type_truncated_cases(self):
        """Test mapping of truncated-type finish reasons."""
        truncated_reasons = ["length", "max_tokens", "token_limit", "max_completion_tokens"]
        for reason in truncated_reasons:
            with self.subTest(reason=reason):
                self.assertEqual(map_finish_reason_to_finish_type(reason), "truncated")

    def test_map_finish_reason_to_finish_type_content_filter_cases(self):
        """Test mapping of content filter-type finish reasons."""
        filter_reasons = ["content_filter", "content_filtered", "safety", "responsible_ai_policy"]
        for reason in filter_reasons:
            with self.subTest(reason=reason):
                self.assertEqual(map_finish_reason_to_finish_type(reason), "content_filter")

    def test_map_finish_reason_to_finish_type_error_cases(self):
        """Test mapping of error-type finish reasons."""
        error_reasons = ["error", "failed", "exception", "timeout", "model_error", "service_unavailable", "rate_limit"]
        for reason in error_reasons:
            with self.subTest(reason=reason):
                self.assertEqual(map_finish_reason_to_finish_type(reason), "error")

    def test_map_finish_reason_to_finish_type_case_insensitive(self):
        """Test that mapping works with different cases."""
        test_cases = [
            ("STOP", "success"),
            ("Stop", "success"),
            ("ERROR", "error"),
            ("Error", "error"),
            ("MAX_TOKENS", "truncated"),
            ("max_tokens", "truncated"),
        ]
        
        for reason, expected in test_cases:
            with self.subTest(reason=reason):
                self.assertEqual(map_finish_reason_to_finish_type(reason), expected)

    def test_map_finish_reason_to_finish_type_pattern_matching(self):
        """Test pattern matching for unmapped finish reasons."""
        test_cases = [
            ("completion_stopped", "success"),  # Contains 'stop'
            ("token_limit_reached", "truncated"),  # Contains 'token' and 'limit'
            ("content_filter_triggered", "content_filter"),  # Contains 'filter'
            ("unexpected_error_occurred", "error"),  # Contains 'error'
            ("service_timeout", "error"),  # Contains 'timeout'
            ("unknown_random_reason", None),  # No matching pattern
        ]
        
        for reason, expected in test_cases:
            with self.subTest(reason=reason):
                self.assertEqual(map_finish_reason_to_finish_type(reason), expected)

    def test_map_finish_reason_to_finish_type_edge_cases(self):
        """Test mapping of edge cases."""
        self.assertEqual(map_finish_reason_to_finish_type(None), None)
        self.assertEqual(map_finish_reason_to_finish_type(""), None)
        self.assertEqual(map_finish_reason_to_finish_type("UNKNOWN_REASON"), None)

    def test_extract_finish_reason_direct_attribute(self):
        """Test extraction when response has direct finish_reason attribute."""
        mock_response = SimpleNamespace(finish_reason="stop")
        
        arguments = {
            "exception": None,
            "result": mock_response
        }
        
        result = extract_finish_reason(arguments)
        self.assertEqual(result, "stop")

    def test_extract_finish_reason_choices_structure(self):
        """Test extraction from choices structure (OpenAI-compatible)."""
        mock_choice = SimpleNamespace(finish_reason="max_tokens")
        mock_response = SimpleNamespace(choices=[mock_choice])
        
        arguments = {
            "exception": None,
            "result": mock_response
        }
        
        result = extract_finish_reason(arguments)
        self.assertEqual(result, "max_tokens")

    def test_extract_finish_reason_additional_kwargs(self):
        """Test extraction from additional_kwargs."""
        mock_response = SimpleNamespace(
            additional_kwargs={
                "finish_reason": "function_call",
                "function_call": {"name": "test"}
            }
        )
        
        arguments = {
            "exception": None,
            "result": mock_response
        }
        
        result = extract_finish_reason(arguments)
        self.assertEqual(result, "function_call")

    def test_extract_finish_reason_response_metadata(self):
        """Test extraction from response_metadata."""
        mock_response = SimpleNamespace(
            response_metadata={
                "stop_reason": "content_filter",
                "model": "gpt-4"
            }
        )
        
        arguments = {
            "exception": None,
            "result": mock_response
        }
        
        result = extract_finish_reason(arguments)
        self.assertEqual(result, "content_filter")

    def test_extract_finish_reason_streaming_response(self):
        """Test extraction from streaming response."""
        mock_response = SimpleNamespace(
            type="stream",
            output_text="Hello, world!"
        )
        
        arguments = {
            "exception": None,
            "result": mock_response
        }
        
        result = extract_finish_reason(arguments)
        self.assertEqual(result, "stop")

    def test_extract_finish_reason_with_exception(self):
        """Test extraction when exception is present."""
        arguments = {
            "exception": Exception("Test error"),
            "result": None
        }
        
        result = extract_finish_reason(arguments)
        self.assertEqual(result, "error")

    def test_extract_finish_reason_fallback_to_status(self):
        """Test fallback to status code when no specific finish reason found."""
        mock_response = SimpleNamespace()  # No finish_reason anywhere
        
        # Mock successful status
        arguments = {
            "exception": None,
            "result": mock_response
        }
        
        # This should fallback to inferring from status
        result = extract_finish_reason(arguments)
        self.assertEqual(result, "stop")  # Default success finish reason

    def test_extract_finish_reason_priority_order(self):
        """Test that extraction follows the correct priority order."""
        # Direct finish_reason should take priority over choices
        mock_choice = SimpleNamespace(finish_reason="max_tokens")
        mock_response = SimpleNamespace(
            finish_reason="stop",
            choices=[mock_choice]  # This should be ignored
        )
        
        arguments = {
            "exception": None,
            "result": mock_response
        }
        
        result = extract_finish_reason(arguments)
        self.assertEqual(result, "stop")

    def test_extract_finish_reason_none_response(self):
        """Test extraction when response is None."""
        arguments = {
            "exception": None,
            "result": None
        }
        
        result = extract_finish_reason(arguments)
        self.assertIsNone(result)

    def test_extract_finish_reason_malformed_response(self):
        """Test extraction with various malformed response structures."""
        test_cases = [
            # Empty response
            SimpleNamespace(),
            
            # Response with empty choices
            SimpleNamespace(choices=[]),
            
            # Response with None choices
            SimpleNamespace(choices=None),
            
            # Response with empty metadata
            SimpleNamespace(response_metadata={}),
            
            # Response with None metadata
            SimpleNamespace(response_metadata=None),
            
            # Response with non-dict metadata
            SimpleNamespace(response_metadata="invalid"),
        ]
        
        for mock_response in test_cases:
            with self.subTest(response=mock_response):
                arguments = {
                    "exception": None,
                    "result": mock_response
                }
                
                # Should not crash and return a default value
                result = extract_finish_reason(arguments)
                self.assertEqual(result, "stop")  # Fallback to success

    def test_integration_extract_and_map(self):
        """Test integration of extract and map functions."""
        test_cases = [
            # (response_structure, expected_finish_reason, expected_finish_type)
            (SimpleNamespace(finish_reason="stop"), "stop", "success"),
            (SimpleNamespace(choices=[SimpleNamespace(finish_reason="max_tokens")]), "max_tokens", "truncated"),
            (SimpleNamespace(additional_kwargs={"stop_reason": "content_filter"}), "content_filter", "content_filter"),
            (SimpleNamespace(response_metadata={"finish_reason": "timeout"}), "timeout", "error"),
        ]
        
        for mock_response, expected_reason, expected_type in test_cases:
            with self.subTest(response=mock_response):
                arguments = {
                    "exception": None,
                    "result": mock_response
                }
                
                # Extract finish reason
                extracted_reason = extract_finish_reason(arguments)
                self.assertEqual(extracted_reason, expected_reason)
                
                # Map to finish type
                mapped_type = map_finish_reason_to_finish_type(extracted_reason)
                self.assertEqual(mapped_type, expected_type)

    def test_real_world_scenarios(self):
        """Test realistic Azure AI Inference response scenarios."""
        
        # Scenario 1: Chat completion with choices
        chat_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content="Hello, how can I help you?",
                        role="assistant"
                    )
                )
            ],
            usage=SimpleNamespace(
                completion_tokens=10,
                prompt_tokens=5,
                total_tokens=15
            )
        )
        
        arguments = {"exception": None, "result": chat_response}
        self.assertEqual(extract_finish_reason(arguments), "stop")
        self.assertEqual(map_finish_reason_to_finish_type("stop"), "success")
        
        # Scenario 2: Token limit reached
        truncated_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="length",
                    message=SimpleNamespace(
                        content="This is a very long response that was truncated...",
                        role="assistant"
                    )
                )
            ]
        )
        
        arguments = {"exception": None, "result": truncated_response}
        self.assertEqual(extract_finish_reason(arguments), "length")
        self.assertEqual(map_finish_reason_to_finish_type("length"), "truncated")
        
        # Scenario 3: Content filter triggered
        filtered_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="content_filter",
                    message=SimpleNamespace(
                        content="I can't help with that request.",
                        role="assistant"
                    )
                )
            ]
        )
        
        arguments = {"exception": None, "result": filtered_response}
        self.assertEqual(extract_finish_reason(arguments), "content_filter")
        self.assertEqual(map_finish_reason_to_finish_type("content_filter"), "content_filter")
        
        # Scenario 4: Streaming response
        stream_response = SimpleNamespace(
            type="stream",
            output_text="This is a streaming response...",
            usage=SimpleNamespace(total_tokens=20)
        )
        
        arguments = {"exception": None, "result": stream_response}
        self.assertEqual(extract_finish_reason(arguments), "stop")
        self.assertEqual(map_finish_reason_to_finish_type("stop"), "success")
        
        # Scenario 5: Tool call completion
        tool_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="tool_calls",
                    message=SimpleNamespace(
                        content="",
                        role="assistant",
                        tool_calls=[{"function": {"name": "get_weather"}}]
                    )
                )
            ]
        )
        
        arguments = {"exception": None, "result": tool_response}
        self.assertEqual(extract_finish_reason(arguments), "tool_calls")
        self.assertEqual(map_finish_reason_to_finish_type("tool_calls"), "tool_call")
        
        # Scenario 6: Azure-specific error
        error_response = SimpleNamespace(
            additional_kwargs={
                "finish_reason": "service_unavailable",
                "error_code": "503"
            }
        )
        
        arguments = {"exception": None, "result": error_response}
        self.assertEqual(extract_finish_reason(arguments), "service_unavailable")
        self.assertEqual(map_finish_reason_to_finish_type("service_unavailable"), "error")


if __name__ == "__main__":
    unittest.main()
