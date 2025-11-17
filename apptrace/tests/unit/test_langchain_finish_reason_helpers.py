"""
Unit tests for LangChain helper functions, particularly finish_reason mapping.
"""
import unittest
from types import SimpleNamespace

from monocle_apptrace.instrumentation.metamodel.finish_types import (
    LANGCHAIN_FINISH_REASON_MAPPING,
)
from monocle_apptrace.instrumentation.metamodel.langchain._helper import (
    extract_finish_reason,
    map_finish_reason_to_finish_type,
)


class TestLangChainFinishReasonHelpers(unittest.TestCase):
    """Test LangChain finish reason extraction and mapping functions."""

    def test_finish_reason_mapping_completeness(self):
        """Test that all expected LangChain finish reasons are mapped."""
        expected_reasons = {
            # Standard completion reasons
            "stop": "success",
            "complete": "success",
            "finished": "success",
            
            # Token limits
            "length": "truncated",
            "max_tokens": "truncated",
            "token_limit": "truncated",
            
            # Tool/function calling
            "tool_calls": "tool_call",
            "function_call": "tool_call",

            # Content filtering and safety
            "content_filter": "content_filter",
            "safety": "content_filter",
            "filtered": "content_filter",

            # Errors
            "error": "error",
            "failed": "error",
            "exception": "error",

            # Provider-specific reasons that might pass through LangChain
            # Anthropic reasons
            "end_turn": "success",
            "stop_sequence": "success",

            # Gemini reasons
            "STOP": "success",
            "SAFETY": "content_filter",
            "RECITATION": "content_filter",
            "OTHER": "error",
        }

        # Check that our mapping contains all expected keys
        for key, value in expected_reasons.items():
            self.assertIn(key, LANGCHAIN_FINISH_REASON_MAPPING)
            self.assertEqual(LANGCHAIN_FINISH_REASON_MAPPING[key], value)

    def test_map_finish_reason_to_finish_type_success_cases(self):
        """Test mapping of success-type finish reasons."""
        success_reasons = ["stop", "complete", "finished", "end_turn", "stop_sequence", "STOP"]
        tool_call_reasons = ["tool_calls", "function_call"]
        for reason in success_reasons:
            with self.subTest(reason=reason):
                self.assertEqual(map_finish_reason_to_finish_type(reason), "success")

        for reason in tool_call_reasons:
            with self.subTest(reason=reason):
                self.assertEqual(map_finish_reason_to_finish_type(reason), "tool_call")

    def test_map_finish_reason_to_finish_type_truncated_cases(self):
        """Test mapping of truncated-type finish reasons."""
        truncated_reasons = ["length", "max_tokens", "token_limit"]
        for reason in truncated_reasons:
            with self.subTest(reason=reason):
                self.assertEqual(map_finish_reason_to_finish_type(reason), "truncated")

    def test_map_finish_reason_to_finish_type_content_filter_cases(self):
        """Test mapping of content filter-type finish reasons."""
        filter_reasons = ["content_filter", "safety", "filtered", "SAFETY", "RECITATION"]
        for reason in filter_reasons:
            with self.subTest(reason=reason):
                self.assertEqual(map_finish_reason_to_finish_type(reason), "content_filter")

    def test_map_finish_reason_to_finish_type_error_cases(self):
        """Test mapping of error-type finish reasons."""
        error_reasons = ["error", "failed", "exception", "OTHER"]
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
            ("safety_filter_triggered", "content_filter"),  # Contains 'filter'
            ("unexpected_error_occurred", "error"),  # Contains 'error'
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

    def test_extract_finish_reason_response_metadata(self):
        """Test extraction from response_metadata."""
        mock_response = SimpleNamespace(
            response_metadata={
                "finish_reason": "max_tokens",
                "model": "gpt-3.5-turbo"
            }
        )

        arguments = {
            "exception": None,
            "result": mock_response
        }

        result = extract_finish_reason(arguments)
        self.assertEqual(result, "max_tokens")

    def test_extract_finish_reason_stop_reason_in_metadata(self):
        """Test extraction of stop_reason from response_metadata (Anthropic style)."""
        mock_response = SimpleNamespace(
            response_metadata={
                "stop_reason": "end_turn",
                "model": "claude-3"
            }
        )

        arguments = {
            "exception": None,
            "result": mock_response
        }

        result = extract_finish_reason(arguments)
        self.assertEqual(result, "end_turn")

    def test_extract_finish_reason_generation_info(self):
        """Test extraction from generation_info."""
        mock_response = SimpleNamespace(
            generation_info={
                "finish_reason": "tool_calls",
                "tokens": 150
            }
        )

        arguments = {
            "exception": None,
            "result": mock_response
        }

        result = extract_finish_reason(arguments)
        self.assertEqual(result, "tool_calls")

    def test_extract_finish_reason_llm_output(self):
        """Test extraction from llm_output (batch responses)."""
        mock_response = SimpleNamespace(
            llm_output={
                "finish_reason": "length",
                "token_usage": {"total": 1000}
            }
        )

        arguments = {
            "exception": None,
            "result": mock_response
        }

        result = extract_finish_reason(arguments)
        self.assertEqual(result, "length")

    def test_extract_finish_reason_additional_kwargs(self):
        """Test extraction from additional_kwargs (AIMessage)."""
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

    def test_extract_finish_reason_generations(self):
        """Test extraction from generations list."""
        mock_generation = SimpleNamespace(
            generation_info={
                "finish_reason": "stop"
            }
        )
        mock_response = SimpleNamespace(
            generations=[mock_generation]
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
        # Direct finish_reason should take priority over metadata
        mock_response = SimpleNamespace(
            finish_reason="stop",
            response_metadata={
                "finish_reason": "max_tokens"  # This should be ignored
            }
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
            (SimpleNamespace(response_metadata={"finish_reason": "max_tokens"}), "max_tokens", "truncated"),
            (SimpleNamespace(generation_info={"stop_reason": "end_turn"}), "end_turn", "success"),
            (SimpleNamespace(additional_kwargs={"finish_reason": "content_filter"}), "content_filter", "content_filter"),
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
        """Test realistic LangChain response scenarios."""

        # Scenario 1: OpenAI-style response through LangChain
        openai_style_response = SimpleNamespace(
            response_metadata={
                "token_usage": {"completion_tokens": 10, "prompt_tokens": 5, "total_tokens": 15},
                "model_name": "gpt-3.5-turbo",
                "finish_reason": "stop"
            }
        )

        arguments = {"exception": None, "result": openai_style_response}
        self.assertEqual(extract_finish_reason(arguments), "stop")
        self.assertEqual(map_finish_reason_to_finish_type("stop"), "success")

        # Scenario 2: Anthropic-style response through LangChain
        anthropic_style_response = SimpleNamespace(
            response_metadata={
                "usage": {"input_tokens": 10, "output_tokens": 20},
                "model": "claude-3-sonnet",
                "stop_reason": "end_turn"
            }
        )

        arguments = {"exception": None, "result": anthropic_style_response}
        self.assertEqual(extract_finish_reason(arguments), "end_turn")
        self.assertEqual(map_finish_reason_to_finish_type("end_turn"), "success")

        # Scenario 3: Tool calling scenario
        tool_call_response = SimpleNamespace(
            additional_kwargs={
                "tool_calls": [{"function": {"name": "search"}}],
                "finish_reason": "tool_calls"
            }
        )

        arguments = {"exception": None, "result": tool_call_response}
        self.assertEqual(extract_finish_reason(arguments), "tool_calls")
        self.assertEqual(map_finish_reason_to_finish_type("tool_calls"), "tool_call")


if __name__ == "__main__":
    unittest.main()