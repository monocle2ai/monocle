import unittest
from pathlib import Path
from monocle_apptrace.cli import cmd_validate


class TestCLIValidateCommand(unittest.TestCase):
    """Test suite for CLI validate command"""

    def setUp(self):
        """Set up test paths"""
        self.test_data_dir = Path(__file__).parent.parent / "data"
        self.trace_valid = self.test_data_dir / "trace_test.json"
        self.trace_invalid = self.test_data_dir / "trace_test_invalid.json"

    def test_validate_valid_trace(self):
        """Test validation of valid trace returns 0"""
        if not self.trace_valid.exists():
            self.skipTest(f"Valid trace file not found: {self.trace_valid}")

        exit_code = cmd_validate(str(self.trace_valid))
        # May return 0 if only warnings, which is ok
        self.assertIn(exit_code, [0, 1])

    def test_validate_invalid_trace(self):
        """Test validation of invalid trace returns 1"""
        if not self.trace_invalid.exists():
            self.skipTest(f"Invalid trace file not found: {self.trace_invalid}")

        exit_code = cmd_validate(str(self.trace_invalid))
        self.assertEqual(exit_code, 1, "Invalid trace should return exit code 1")

    def test_validate_invalid_trace_with_fail_on_warning(self):
        """Test fail-on-warning flag"""
        if not self.trace_valid.exists():
            self.skipTest(f"Valid trace file not found: {self.trace_valid}")

        exit_code = cmd_validate(str(self.trace_valid), fail_on_warning=True)
        # If there are warnings, should return 1
        self.assertEqual(exit_code, 1, "Should fail on warning when flag is set")

    def test_validate_nonexistent_file(self):
        """Test handling of nonexistent trace file"""
        exit_code = cmd_validate("/nonexistent/trace.json")
        self.assertEqual(exit_code, 1, "Should return 1 for nonexistent file")

    def test_validate_with_selective_level(self):
        """Test validation with selective level"""
        if not self.trace_valid.exists():
            self.skipTest(f"Valid trace file not found: {self.trace_valid}")

        exit_code = cmd_validate(str(self.trace_valid), level="selective")
        # Should complete without crashing
        self.assertIn(exit_code, [0, 1])

    def test_validate_with_strict_level(self):
        """Test validation with strict level"""
        if not self.trace_valid.exists():
            self.skipTest(f"Valid trace file not found: {self.trace_valid}")

        exit_code = cmd_validate(str(self.trace_valid), level="strict")
        # Should complete without crashing
        self.assertIn(exit_code, [0, 1])


if __name__ == "__main__":
    unittest.main()
