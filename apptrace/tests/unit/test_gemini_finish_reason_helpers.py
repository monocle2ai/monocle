"""
Unit tests for Gemini helper functions, particularly finish_reason mapping.
"""
import unittest
from types import SimpleNamespace

from monocle_apptrace.instrumentation.metamodel.finish_types import (
    GEMINI_FINISH_REASON_MAPPING,
)
from monocle_apptrace.instrumentation.metamodel.gemini._helper import (
    extract_finish_reason,
    map_finish_reason_to_finish_type,
)


class TestGeminiFinishReasonHelpers(unittest.TestCase):
    """Test Gemini finish reason extraction and mapping functions."""

    def test_finish_reason_mapping_completeness(self):
        """Test that all expected Gemini finish reasons are mapped."""
        expected_reasons = {
            "STOP": "success",
            "FUNCTION_CALL":"tool_call",
            "MAX_TOKENS": "truncated",
            "SAFETY": "content_filter",
            "RECITATION": "content_filter",
            "MALFORMED_FUNCTION_CALL": "tool_call_error",
            "OTHER": "error",
            "FINISH_REASON_UNSPECIFIED": None
        }
        
        self.assertEqual(GEMINI_FINISH_REASON_MAPPING, expected_reasons)

    def test_map_finish_reason_to_finish_type_success_cases(self):
        """Test mapping of success-type finish reasons."""
        self.assertEqual(map_finish_reason_to_finish_type("STOP"), "success")

    def test_map_finish_reason_to_finish_type_truncated_cases(self):
        """Test mapping of truncated-type finish reasons."""
        self.assertEqual(map_finish_reason_to_finish_type("MAX_TOKENS"), "truncated")

    def test_map_finish_reason_to_finish_type_content_filter_cases(self):
        """Test mapping of content filter-type finish reasons."""
        self.assertEqual(map_finish_reason_to_finish_type("SAFETY"), "content_filter")
        self.assertEqual(map_finish_reason_to_finish_type("RECITATION"), "content_filter")

    def test_map_finish_reason_to_finish_type_error_cases(self):
        """Test mapping of error-type finish reasons."""
        self.assertEqual(map_finish_reason_to_finish_type("OTHER"), "error")

    def test_map_finish_reason_to_finish_type_tool_call_error_cases(self):
        """Test mapping of tool call error-type finish reasons."""
        self.assertEqual(map_finish_reason_to_finish_type("MALFORMED_FUNCTION_CALL"), "tool_call_error")

    def test_map_finish_reason_to_finish_type_unspecified(self):
        """Test mapping of unspecified finish reason."""
        self.assertEqual(map_finish_reason_to_finish_type("FINISH_REASON_UNSPECIFIED"), None)

    def test_map_finish_reason_to_finish_type_unknown(self):
        """Test mapping of unknown finish reasons."""
        self.assertEqual(map_finish_reason_to_finish_type("UNKNOWN_REASON"), None)
        self.assertEqual(map_finish_reason_to_finish_type(""), None)
        self.assertEqual(map_finish_reason_to_finish_type(None), None)

    def test_extract_finish_reason_success(self):
        """Test successful extraction of finish reason from Gemini response."""
        # Mock Gemini response structure
        mock_candidate = SimpleNamespace(finish_reason="STOP")
        mock_response = SimpleNamespace(candidates=[mock_candidate])
        
        arguments = {
            "exception": None,
            "result": mock_response
        }
        
        result = extract_finish_reason(arguments)
        self.assertEqual(result, "STOP")

    def test_extract_finish_reason_with_exception(self):
        """Test extraction when exception is present."""
        arguments = {
            "exception": Exception("Test error"),
            "result": None
        }
        
        result = extract_finish_reason(arguments)
        self.assertIsNone(result)

    def test_extract_finish_reason_no_candidates(self):
        """Test extraction when response has no candidates."""
        mock_response = SimpleNamespace(candidates=[])
        
        arguments = {
            "exception": None,
            "result": mock_response
        }
        
        result = extract_finish_reason(arguments)
        self.assertIsNone(result)

    def test_extract_finish_reason_no_finish_reason_attribute(self):
        """Test extraction when candidate has no finish_reason attribute."""
        mock_candidate = SimpleNamespace()  # No finish_reason attribute
        mock_response = SimpleNamespace(candidates=[mock_candidate])
        
        arguments = {
            "exception": None,
            "result": mock_response
        }
        
        result = extract_finish_reason(arguments)
        self.assertIsNone(result)

    def test_extract_finish_reason_multiple_candidates(self):
        """Test extraction when response has multiple candidates (uses first one)."""
        mock_candidate1 = SimpleNamespace(finish_reason="STOP")
        mock_candidate2 = SimpleNamespace(finish_reason="MAX_TOKENS")
        mock_response = SimpleNamespace(candidates=[mock_candidate1, mock_candidate2])
        
        arguments = {
            "exception": None,
            "result": mock_response
        }
        
        result = extract_finish_reason(arguments)
        self.assertEqual(result, "STOP")  # Should use first candidate

    def test_extract_finish_reason_various_reasons(self):
        """Test extraction of various finish reasons."""
        finish_reasons = ["STOP", "MAX_TOKENS", "SAFETY", "RECITATION", "MALFORMED_FUNCTION_CALL", "OTHER", "FINISH_REASON_UNSPECIFIED"]
        
        for reason in finish_reasons:
            with self.subTest(reason=reason):
                mock_candidate = SimpleNamespace(finish_reason=reason)
                mock_response = SimpleNamespace(candidates=[mock_candidate])
                
                arguments = {
                    "exception": None,
                    "result": mock_response
                }
                
                result = extract_finish_reason(arguments)
                self.assertEqual(result, reason)

    def test_extract_finish_reason_none_response(self):
        """Test extraction when response is None."""
        arguments = {
            "exception": None,
            "result": None
        }
        
        result = extract_finish_reason(arguments)
        self.assertIsNone(result)

    def test_extract_finish_reason_malformed_response(self):
        """Test extraction with malformed response structure."""
        # Response without candidates attribute
        mock_response = SimpleNamespace()
        
        arguments = {
            "exception": None,
            "result": mock_response
        }
        
        result = extract_finish_reason(arguments)
        self.assertIsNone(result)

    def test_integration_extract_and_map(self):
        """Test integration of extract and map functions."""
        test_cases = [
            ("STOP", "success"),
            ("MAX_TOKENS", "truncated"),
            ("SAFETY", "content_filter"),
            ("RECITATION", "content_filter"),
            ("MALFORMED_FUNCTION_CALL", "tool_call_error"),
            ("OTHER", "error"),
            ("FINISH_REASON_UNSPECIFIED", None),
        ]
        
        for finish_reason, expected_finish_type in test_cases:
            with self.subTest(finish_reason=finish_reason):
                # Create mock response
                mock_candidate = SimpleNamespace(finish_reason=finish_reason)
                mock_response = SimpleNamespace(candidates=[mock_candidate])
                
                arguments = {
                    "exception": None,
                    "result": mock_response
                }
                
                # Extract finish reason
                extracted_reason = extract_finish_reason(arguments)
                self.assertEqual(extracted_reason, finish_reason)
                
                # Map to finish type
                mapped_type = map_finish_reason_to_finish_type(extracted_reason)
                self.assertEqual(mapped_type, expected_finish_type)


if __name__ == "__main__":
    unittest.main()
