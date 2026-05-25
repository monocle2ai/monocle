from abc import ABC, abstractmethod
from typing import List, Dict, Any
import re


class ValidationError:
    """Represents a validation error/warning"""
    def __init__(self, field: str, span_name: str, message: str, severity: str = "error"):
        self.field = field
        self.span_name = span_name
        self.message = message
        self.severity = severity  # "error" ou "warning"

    def __str__(self):
        return f"{self.severity.upper()}: {self.message}"


class Rule(ABC):
    """Base class for validation rules"""

    @abstractmethod
    def validate(self, span: Dict[str, Any]) -> List[ValidationError]:
        """Validate a span and return list of errors/warnings"""
        pass


class RequiredFieldRule(Rule):
    """Validate that required fields are present"""

    def __init__(self, span_type: str, required_fields: List[str]):
        self.span_type = span_type
        self.required_fields = required_fields

    def validate(self, span: Dict[str, Any]) -> List[ValidationError]:
        # Ne valider que si c'est le bon type de span
        attrs = span.get("attributes", {})
        if attrs.get("span.type") != self.span_type:
            return []

        errors = []
        for field in self.required_fields:
            if field not in attrs or attrs[field] is None:
                errors.append(ValidationError(
                    field=field,
                    span_name=span.get("name", "unknown"),
                    message=f"missing required field {field}",
                    severity="error"
                ))
        return errors


class TokenCountRule(Rule):
    """Validate token counts in metadata"""

    def validate(self, span: Dict[str, Any]) -> List[ValidationError]:
        attrs = span.get("attributes", {})
        if attrs.get("span.type") != "inference":
            return []

        warnings = []
        events = span.get("events", [])
        metadata_event = next((e for e in events if e.get("name") == "metadata"), None)

        if not metadata_event:
            return []

        metadata_attrs = metadata_event.get("attributes", {})

        # Vérifier si provider retourne des tokens
        provider = attrs.get("entity.1.provider_name", "unknown")
        providers_with_tokens = ["openai", "anthropic", "gemini", "bedrock", "azure"]

        if provider in providers_with_tokens:
            if "prompt_tokens" not in metadata_attrs:
                warnings.append(ValidationError(
                    field="metadata.prompt_tokens",
                    span_name=span.get("name", "unknown"),
                    message="missing prompt_tokens in metadata",
                    severity="error"
                ))
            if "completion_tokens" not in metadata_attrs:
                warnings.append(ValidationError(
                    field="metadata.completion_tokens",
                    span_name=span.get("name", "unknown"),
                    message="missing completion_tokens in metadata",
                    severity="error"
                ))
        else:
            # Provider qui ne retourne pas de tokens = warning seulement
            if "prompt_tokens" not in metadata_attrs:
                warnings.append(ValidationError(
                    field="metadata.prompt_tokens",
                    span_name=span.get("name", "unknown"),
                    message="missing prompt_tokens in metadata (optional for this provider)",
                    severity="warning"
                ))

        return warnings


class ToolMetadataRule(Rule):
    """Validate tool call metadata"""

    def validate(self, span: Dict[str, Any]) -> List[ValidationError]:
        attrs = span.get("attributes", {})
        if attrs.get("span.type") != "agentic.tool.invocation":
            return []

        errors = []
        events = span.get("events", [])
        metadata_event = next((e for e in events if e.get("name") == "metadata"), None)

        if not metadata_event:
            errors.append(ValidationError(
                field="metadata",
                span_name=span.get("name", "unknown"),
                message="missing metadata event in tool invocation",
                severity="error"
            ))
            return errors

        metadata_attrs = metadata_event.get("attributes", {})
        required_tool_fields = ["tool.status", "tool.return_value"]

        for field in required_tool_fields:
            if field not in metadata_attrs:
                errors.append(ValidationError(
                    field=field,
                    span_name=span.get("name", "unknown"),
                    message=f"missing {field} in tool metadata",
                    severity="error"
                ))

        return errors


class NamingConventionRule(Rule):
    """Validate span naming conventions"""

    def validate(self, span: Dict[str, Any]) -> List[ValidationError]:
        span_name = span.get("name", "")

        # Span names should be snake_case
        if not self._is_valid_name(span_name):
            return [ValidationError(
                field="name",
                span_name=span_name,
                message=f"span name should be snake_case, got '{span_name}'",
                severity="warning"
            )]

        return []

    @staticmethod
    def _is_valid_name(name: str) -> bool:
        """Check if name follows snake_case convention"""
        if not name:
            return False
        # Allow snake_case, dots, and underscores
        return bool(re.match(r'^[a-z0-9._]+$', name))
