from typing import List
from monocle_apptrace.linter.rules import ValidationError
from monocle_apptrace.linter.validator import ValidationResult


class ValidationReporter:
    """Format validation results for human-readable output.

    Converts validation results into formatted messages suitable for CLI output
    and determines appropriate exit codes based on validation status.

    Example:
        >>> reporter = ValidationReporter()
        >>> output = reporter.format_results(results)
        >>> print(output)
        ERROR: missing required field entity.2.name
        WARNING: missing completion_tokens in metadata
    """

    @staticmethod
    def format_results(results: ValidationResult, fail_on_warning: bool = False) -> str:
        """Format validation results as ERROR/WARNING messages.

        Converts validation errors and warnings into human-readable format
        suitable for console output.

        Args:
            results (ValidationResult): Validation results to format
            fail_on_warning (bool): If True, warnings are treated as errors
                (affects display only, not the actual exit code)

        Returns:
            str: Formatted output with ERROR and WARNING lines, one per line

        Example:
            >>> output = ValidationReporter.format_results(results)
            >>> print(output)
            ERROR: missing required field entity.2.name
            WARNING: span name should be snake_case, got 'OpenAI_Create'
        """
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
        """Determine exit code based on validation results.

        Returns 0 if validation passes (no errors, or no errors and
        fail_on_warning is False). Returns 1 if there are errors or
        if there are warnings and fail_on_warning is True.

        Args:
            results (ValidationResult): Validation results
            fail_on_warning (bool): If True, warnings cause non-zero exit code

        Returns:
            int: Exit code (0 for success, 1 for failure)

        Example:
            >>> exit_code = ValidationReporter.get_exit_code(results, fail_on_warning=True)
            >>> sys.exit(exit_code)
        """
        if results.has_errors():
            return 1

        if results.has_warnings() and fail_on_warning:
            return 1

        return 0
