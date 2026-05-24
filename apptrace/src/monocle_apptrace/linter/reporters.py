from typing import List
from monocle_apptrace.linter.rules import ValidationError
from monocle_apptrace.linter.validator import ValidationResult


class ValidationReporter:
    """Format validation results for output"""

    @staticmethod
    def format_results(results: ValidationResult, fail_on_warning: bool = False) -> str:
        """Format validation results as ERROR/WARNING messages"""
        output = []

        # Add errors
        for error in results.all_errors:
            output.append(f"ERROR: {error.message}")

        # Add warnings
        for warning in results.all_warnings:
            output.append(f"WARNING: {warning.message}")

        # If no errors or warnings
        if not output:
            output.append("All spans are valid!")

        return "\n".join(output)

    @staticmethod
    def get_exit_code(results: ValidationResult, fail_on_warning: bool = False) -> int:
        """Determine exit code based on results"""
        if results.has_errors():
            return 1

        if results.has_warnings() and fail_on_warning:
            return 1

        return 0
