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
    """Contains validation results"""
    def __init__(self, errors: List[ValidationError]):
        self.errors = errors
        self.all_errors = [e for e in errors if e.severity == "error"]
        self.all_warnings = [e for e in errors if e.severity == "warning"]

    def has_errors(self) -> bool:
        return len(self.all_errors) > 0

    def has_warnings(self) -> bool:
        return len(self.all_warnings) > 0


class MonocleValidator:
    """Main validator class"""

    def __init__(self):
        """Initialize validator with specs and rules"""
        self.specs = SpecsLoader.load_specs()
        self.rules = self._build_rules()

    def _build_rules(self) -> List[Rule]:
        """Build validation rules from specs"""
        rules = []

        # Rule 1: Required fields for inference spans
        rules.append(RequiredFieldRule(
            span_type="inference",
            required_fields=["entity.2.name", "entity.2.type"]
        ))

        # Rule 2: Token counts
        rules.append(TokenCountRule())

        # Rule 3: Tool metadata
        rules.append(ToolMetadataRule())

        # Rule 4: Naming convention
        rules.append(NamingConventionRule())

        return rules

    def validate_trace_file(self, trace_file: Path) -> ValidationResult:
        """Validate entire trace file"""
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
        """Validate a single span"""
        errors = []
        for rule in self.rules:
            errors.extend(rule.validate(span))
        return errors
