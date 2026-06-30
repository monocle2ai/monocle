import json
from pathlib import Path
from typing import List, Dict, Any

from monocle_apptrace.linter.specs_loader import SpecsLoader
from monocle_apptrace.linter.rules import (
    Rule,
    RequiredFieldRule,
    TokenCountRule,
    ToolMetadataRule,
    NamingConventionRule,
    ValidationError
)


class ValidationResult:
    """Contains validation results from trace validation.

    Separates errors and warnings, and provides convenience methods to check
    if there are any validation issues.

    Attributes:
        errors (List[ValidationError]): All validation issues (errors and warnings)
        all_errors (List[ValidationError]): Only validation errors
        all_warnings (List[ValidationError]): Only validation warnings

    Example:
        >>> results = validator.validate_trace_file(Path("trace.json"))
        >>> if results.has_errors():
        ...     print(f"Found {len(results.all_errors)} errors")
    """

    def __init__(self, errors: List[ValidationError]):
        """Initialize validation results.

        Args:
            errors (List[ValidationError]): List of all validation issues
        """
        self.errors = errors
        self.all_errors = [e for e in errors if e.severity == "error"]
        self.all_warnings = [e for e in errors if e.severity == "warning"]

    def has_errors(self) -> bool:
        """Check if there are any validation errors.

        Returns:
            bool: True if errors exist, False otherwise
        """
        return len(self.all_errors) > 0

    def has_warnings(self) -> bool:
        """Check if there are any validation warnings.

        Returns:
            bool: True if warnings exist, False otherwise
        """
        return len(self.all_warnings) > 0


class MonocleValidator:
    """Main validator for Monocle trace conformance.

    Validates traces against the Monocle metamodel to ensure they follow
    observability conventions. Uses validation rules to check:
    - Required attributes on spans
    - Token counts in inference spans
    - Metadata in tool invocations
    - Naming conventions

    Attributes:
        specs (Dict[str, Any]): Loaded specifications from monocle-specs
        rules (List[Rule]): List of validation rules to apply

    Example:
        >>> validator = MonocleValidator()
        >>> results = validator.validate_trace_file(Path("trace.json"))
        >>> if results.has_errors():
        ...     print(f"Found {len(results.all_errors)} validation errors")
    """

    def __init__(self):
        """Initialize validator with specs and rules.

        Loads validation specifications from monocle-specs repository and
        builds the default set of validation rules.
        """
        self.specs = SpecsLoader.load_specs()
        self.rules = self._build_rules()

    def _build_rules(self) -> List[Rule]:
        """Build validation rules from specifications.

        Creates all validation rules that will be applied to spans:
        1. RequiredFieldRule - Checks required attributes
        2. TokenCountRule - Validates token counts in metadata
        3. ToolMetadataRule - Validates tool invocation metadata
        4. NamingConventionRule - Enforces naming conventions

        Returns:
            List[Rule]: List of validation rules
        """
        rules = []

        # Rule 1: Required fields for inference spans
        rules.append(RequiredFieldRule(
            span_type="inference",
            required_fields=["entity.2.name", "entity.2.type"]
        ))

        # Rule 2: Token counts
        rules.append(TokenCountRule())

        # Rule 3: Tool metadata
        ### Temporarily skipping tool metadata rule. Need to refactor as a follow up patch.
        ## rules.append(ToolMetadataRule())

        # Rule 4: Naming convention
        rules.append(NamingConventionRule())

        return rules

    def validate_trace_file(self, trace_file: Path) -> ValidationResult:
        """Validate an entire trace JSON file.

        Reads a trace file, parses the JSON, and validates all spans against
        the defined validation rules.

        Args:
            trace_file (Path): Path to the trace JSON file

        Returns:
            ValidationResult: Contains all validation errors and warnings

        Raises:
            FileNotFoundError: If trace file doesn't exist
            ValueError: If JSON in trace file is invalid

        Example:
            >>> validator = MonocleValidator()
            >>> results = validator.validate_trace_file(Path("trace.json"))
            >>> print(f"Errors: {len(results.all_errors)}")
            >>> print(f"Warnings: {len(results.all_warnings)}")
        """
        if not trace_file.exists():
            raise FileNotFoundError(f"Trace file not found: {trace_file}")

        # Load trace JSON
        with open(trace_file) as f:
            try:
                traces = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in trace file: {e}")

        # Ensure traces is a list
        if isinstance(traces, dict):
            traces = [traces]

        # Validate each span
        all_errors = []
        for span in traces:
            all_errors.extend(self.validate_span(span))

        return ValidationResult(all_errors)

    def validate_span(self, span: Dict[str, Any]) -> List[ValidationError]:
        """Validate a single span against all validation rules.

        Applies each validation rule to the span and collects any errors
        or warnings found.

        Args:
            span (Dict[str, Any]): Span object to validate with structure:
                {
                    "name": "span_name",
                    "attributes": {...},
                    "events": [...]
                }

        Returns:
            List[ValidationError]: List of validation issues (errors + warnings)

        Example:
            >>> span = {"name": "test", "attributes": {...}, "events": [...]}
            >>> errors = validator.validate_span(span)
            >>> for error in errors:
            ...     print(f"{error.severity}: {error.message}")
        """
        errors = []
        for rule in self.rules:
            errors.extend(rule.validate(span))
        return errors
