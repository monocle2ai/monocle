import unittest
from pathlib import Path
from monocle_apptrace.linter import MonocleValidator, ValidationResult


class TestMonocleValidator(unittest.TestCase):
    """Test suite for MonocleValidator"""

    def setUp(self):
        """Initialize validator for each test"""
        self.validator = MonocleValidator()

    def test_validate_valid_inference_span(self):
        """Test validation of a valid inference span"""
        span = {
            "name": "openai.create",
            "attributes": {
                "span.type": "inference",
                "entity.2.name": "gpt-4",
                "entity.2.type": "model.llm"
            },
            "events": [
                {
                    "name": "metadata",
                    "attributes": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5
                    }
                }
            ]
        }

        errors = self.validator.validate_span(span)
        self.assertEqual(len(errors), 0, "Valid span should have no errors")

    def test_validate_missing_entity_name(self):
        """Test detection of missing entity.2.name"""
        span = {
            "name": "inference_call",
            "attributes": {
                "span.type": "inference",
                "entity.2.type": "model.llm"
            },
            "events": []
        }

        errors = self.validator.validate_span(span)
        error_messages = [e.message for e in errors]
        self.assertIn("missing required field entity.2.name", error_messages)

    def test_validate_missing_token_counts(self):
        """Test detection of missing token counts"""
        span = {
            "name": "inference_call",
            "attributes": {
                "span.type": "inference",
                "entity.1.provider_name": "openai",
                "entity.2.name": "gpt-4",
                "entity.2.type": "model.llm"
            },
            "events": [
                {
                    "name": "metadata",
                    "attributes": {}
                }
            ]
        }

        errors = self.validator.validate_span(span)
        error_messages = [e.message for e in errors]
        self.assertTrue(
            any("prompt_tokens" in msg for msg in error_messages),
            "Should detect missing prompt_tokens"
        )

    def test_validate_tool_metadata_missing(self):
        """Test detection of missing tool metadata"""
        span = {
            "name": "search_tool",
            "attributes": {
                "span.type": "agentic.tool.invocation",
                "entity.3.name": "search"
            },
            "events": [
                {
                    "name": "metadata",
                    "attributes": {}
                }
            ]
        }

        errors = self.validator.validate_span(span)
        error_messages = [e.message for e in errors]
        self.assertTrue(
            any("tool.status" in msg for msg in error_messages),
            "Should detect missing tool.status"
        )

    def test_validate_naming_convention(self):
        """Test detection of invalid span naming"""
        span = {
            "name": "OpenAI_ChatCompletion_Create",  # Invalid: camelCase
            "attributes": {
                "span.type": "inference",
                "entity.2.name": "gpt-4",
                "entity.2.type": "model.llm"
            },
            "events": [
                {
                    "name": "metadata",
                    "attributes": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5
                    }
                }
            ]
        }

        errors = self.validator.validate_span(span)
        warnings = [e for e in errors if e.severity == "warning"]
        self.assertTrue(
            any("snake_case" in w.message for w in warnings),
            "Should warn about snake_case naming"
        )

    def test_validate_trace_file(self):
        """Test validation of entire trace file"""
        trace_file = Path(__file__).parent.parent / "data" / "trace_test.json"
        if not trace_file.exists():
            self.skipTest(f"Trace file not found: {trace_file}")

        results = self.validator.validate_trace_file(trace_file)
        self.assertIsInstance(results, ValidationResult)
        # Should have some results (at least the naming warning)
        self.assertGreater(len(results.errors), 0)


if __name__ == "__main__":
    unittest.main()
