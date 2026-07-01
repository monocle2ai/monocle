"""
Unit tests for Monocle feedback helper functions.
"""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from monocle_apptrace.instrumentation.metamodel.feedback._helper import (
    extract_session_id,
    extract_turn_id,
    extract_feedback_string,
)


class TestFeedbackHelpers(unittest.TestCase):
    """Test feedback extraction helper functions."""

    # ========================================================================
    # Tests for extract_session_id
    # ========================================================================

    def test_extract_session_id_from_kwargs(self):
        """Test extracting session_id from kwargs."""
        arguments = {
            "kwargs": {"session_id": "session_123"},
            "args": [],
            "result": None,
        }
        self.assertEqual(extract_session_id(arguments), "session_123")

    def test_extract_session_id_from_result_attribute(self):
        """Test extracting session_id from result object attribute."""
        result_obj = SimpleNamespace(session_id="session_456")
        arguments = {
            "kwargs": {},
            "args": [],
            "result": result_obj,
        }
        self.assertEqual(extract_session_id(arguments), "session_456")

    def test_extract_session_id_from_result_dict(self):
        """Test extracting session_id from result dict."""
        arguments = {
            "kwargs": {},
            "args": [],
            "result": {"session_id": "session_789"},
        }
        self.assertEqual(extract_session_id(arguments), "session_789")

    def test_extract_session_id_prefers_kwargs(self):
        """Test that kwargs is preferred over result."""
        result_obj = SimpleNamespace(session_id="result_session")
        arguments = {
            "kwargs": {"session_id": "kwargs_session"},
            "args": [],
            "result": result_obj,
        }
        self.assertEqual(extract_session_id(arguments), "kwargs_session")

    def test_extract_session_id_type_conversion(self):
        """Test that session_id is converted to string."""
        arguments = {
            "kwargs": {"session_id": 12345},
            "args": [],
            "result": None,
        }
        self.assertEqual(extract_session_id(arguments), "12345")

    def test_extract_session_id_missing(self):
        """Test handling when session_id is not found."""
        arguments = {
            "kwargs": {},
            "args": [],
            "result": None,
        }
        with patch('monocle_apptrace.instrumentation.metamodel.feedback._helper.logger') as mock_logger:
            result = extract_session_id(arguments)
            self.assertIsNone(result)
            mock_logger.warning.assert_called()

    def test_extract_session_id_exception_handling(self):
        """Test exception handling in extract_session_id."""
        arguments = None  # Will cause exception
        with patch('monocle_apptrace.instrumentation.metamodel.feedback._helper.logger') as mock_logger:
            result = extract_session_id(arguments)
            self.assertIsNone(result)
            mock_logger.warning.assert_called()

    # ========================================================================
    # Tests for extract_turn_id
    # ========================================================================

    def test_extract_turn_id_from_kwargs(self):
        """Test extracting turn_id from kwargs."""
        arguments = {
            "kwargs": {"turn_id": "turn_1"},
            "args": [],
            "result": None,
        }
        self.assertEqual(extract_turn_id(arguments), "turn_1")

    def test_extract_turn_id_from_result_attribute(self):
        """Test extracting turn_id from result object attribute."""
        result_obj = SimpleNamespace(turn_id="turn_2")
        arguments = {
            "kwargs": {},
            "args": [],
            "result": result_obj,
        }
        self.assertEqual(extract_turn_id(arguments), "turn_2")

    def test_extract_turn_id_from_result_dict(self):
        """Test extracting turn_id from result dict."""
        arguments = {
            "kwargs": {},
            "args": [],
            "result": {"turn_id": "turn_3"},
        }
        self.assertEqual(extract_turn_id(arguments), "turn_3")

    def test_extract_turn_id_prefers_kwargs(self):
        """Test that kwargs is preferred over result for turn_id."""
        result_obj = SimpleNamespace(turn_id="result_turn")
        arguments = {
            "kwargs": {"turn_id": "kwargs_turn"},
            "args": [],
            "result": result_obj,
        }
        self.assertEqual(extract_turn_id(arguments), "kwargs_turn")

    def test_extract_turn_id_type_conversion(self):
        """Test that turn_id is converted to string."""
        arguments = {
            "kwargs": {"turn_id": 999},
            "args": [],
            "result": None,
        }
        self.assertEqual(extract_turn_id(arguments), "999")

    def test_extract_turn_id_missing_returns_none(self):
        """Test that missing turn_id returns None (it's optional)."""
        arguments = {
            "kwargs": {},
            "args": [],
            "result": None,
        }
        result = extract_turn_id(arguments)
        self.assertIsNone(result)

    def test_extract_turn_id_exception_handling(self):
        """Test exception handling in extract_turn_id."""
        arguments = None  # Will cause exception
        with patch('monocle_apptrace.instrumentation.metamodel.feedback._helper.logger') as mock_logger:
            result = extract_turn_id(arguments)
            self.assertIsNone(result)
            mock_logger.warning.assert_called()

    # ========================================================================
    # Tests for extract_feedback_string
    # ========================================================================

    def test_extract_feedback_string_from_result_attribute(self):
        """Test extracting feedback from result object attribute."""
        result_obj = SimpleNamespace(feedback="Great response!")
        arguments = {
            "kwargs": {},
            "args": [],
            "result": result_obj,
        }
        with patch('monocle_apptrace.instrumentation.metamodel.feedback._helper.get_status_code', return_value='success'):
            self.assertEqual(extract_feedback_string(arguments), "Great response!")

    def test_extract_feedback_string_from_result_dict(self):
        """Test extracting feedback from result dict."""
        arguments = {
            "kwargs": {},
            "args": [],
            "result": {"feedback": "Helpful answer"},
        }
        with patch('monocle_apptrace.instrumentation.metamodel.feedback._helper.get_status_code', return_value='success'):
            self.assertEqual(extract_feedback_string(arguments), "Helpful answer")

    def test_extract_feedback_string_from_string_result(self):
        """Test extracting feedback when result is directly a string."""
        arguments = {
            "kwargs": {},
            "args": [],
            "result": "Direct feedback string",
        }
        with patch('monocle_apptrace.instrumentation.metamodel.feedback._helper.get_status_code', return_value='success'):
            self.assertEqual(extract_feedback_string(arguments), "Direct feedback string")

    def test_extract_feedback_string_type_conversion(self):
        """Test that feedback is converted to string."""
        result_obj = SimpleNamespace(feedback=12345)
        arguments = {
            "kwargs": {},
            "args": [],
            "result": result_obj,
        }
        with patch('monocle_apptrace.instrumentation.metamodel.feedback._helper.get_status_code', return_value='success'):
            self.assertEqual(extract_feedback_string(arguments), "12345")

    def test_extract_feedback_string_non_success_status(self):
        """Test that non-success status returns None."""
        result_obj = SimpleNamespace(feedback="This should be ignored")
        arguments = {
            "kwargs": {},
            "args": [],
            "result": result_obj,
        }
        # Patch where get_status_code is used, not where it's defined
        with patch('monocle_apptrace.instrumentation.metamodel.feedback._helper.get_status_code', return_value='error'):
            result = extract_feedback_string(arguments)
            self.assertIsNone(result)

    def test_extract_feedback_string_missing_returns_none(self):
        """Test that missing feedback returns None."""
        arguments = {
            "kwargs": {},
            "args": [],
            "result": {"other_field": "value"},
        }
        with patch('monocle_apptrace.instrumentation.metamodel.feedback._helper.get_status_code', return_value='success'):
            result = extract_feedback_string(arguments)
            self.assertIsNone(result)

    def test_extract_feedback_string_exception_handling(self):
        """Test exception handling in extract_feedback_string."""
        arguments = None  # Will cause exception
        with patch('monocle_apptrace.instrumentation.metamodel.feedback._helper.logger') as mock_logger:
            result = extract_feedback_string(arguments)
            self.assertIsNone(result)
            mock_logger.warning.assert_called()

    def test_extract_feedback_string_multiline(self):
        """Test extracting multiline feedback strings."""
        multiline_feedback = """This is a detailed feedback
spanning multiple lines
with various content"""
        arguments = {
            "kwargs": {},
            "args": [],
            "result": multiline_feedback,
        }
        with patch('monocle_apptrace.instrumentation.metamodel.feedback._helper.get_status_code', return_value='success'):
            self.assertEqual(extract_feedback_string(arguments), multiline_feedback)

    def test_extract_feedback_string_empty(self):
        """Test handling empty feedback string - returns None since empty is treated as missing."""
        arguments = {
            "kwargs": {},
            "args": [],
            "result": {"feedback": ""},
        }
        with patch('monocle_apptrace.instrumentation.metamodel.feedback._helper.get_status_code', return_value='success'):
            # Empty string is treated as missing and returns None
            self.assertIsNone(extract_feedback_string(arguments))

    def test_extract_feedback_string_unicode(self):
        """Test extracting feedback with unicode characters."""
        unicode_feedback = "Great job! 👍 很好 ありがとう"
        arguments = {
            "kwargs": {},
            "args": [],
            "result": {"feedback": unicode_feedback},
        }
        with patch('monocle_apptrace.instrumentation.metamodel.feedback._helper.get_status_code', return_value='success'):
            self.assertEqual(extract_feedback_string(arguments), unicode_feedback)


if __name__ == '__main__':
    unittest.main()
