# Copyright (C) Http Inc 2023-2024. All rights reserved

import json
import logging
from typing import Any, Dict, List, Optional, Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

logger = logging.getLogger(__name__)

class MockSpanExporter(SpanExporter):
    """
    Comprehensive mock span exporter for testing that combines span capture and validation.
    
    Features:
    - Captures spans without making HTTP calls (prevents connection errors)
    - Stores spans for test assertions and data retrieval
    - Validates span attributes with configurable strict/non-strict mode
    - Context manager support for automatic cleanup
    """
    
    def __init__(self, endpoint: Optional[str] = None, strict_mode: bool = True):
        """
        Initialize mock exporter.
        
        Args:
            endpoint: Mock endpoint URL for compatibility
            strict_mode: If True, attribute validation failures raise AssertionError
        """
        self.endpoint = endpoint or "mock://localhost:3000/api/v1/traces"
        self.exported_spans: List[Dict[str, Any]] = []
        self._closed = False
        
        # Attribute validation features (from MockExporter)
        self.attributes_to_check: Dict[str, str] = {}
        self.strict_mode = strict_mode
        self.current_trace_id: Optional[int] = None
        self.current_file_path: Optional[str] = None
    
    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Export spans by storing them in memory and optionally validating attributes."""
        if self._closed:
            logger.debug("Exporter already shutdown, ignoring export")
            return SpanExportResult.FAILURE
            
        span_list = {"batch": []}
        
        for span in spans:
            # Validate attributes if checks are configured
            self._validate_span_attributes(span)
            
            # Convert span to JSON format (same as HttpSpanExporter)
            obj = json.loads(span.to_json())
            if obj["parent_id"] is None:
                obj["parent_id"] = "None"
            else:
                obj["parent_id"] = self._remove_0x_from_start(obj["parent_id"])
            if obj["context"] is not None:
                obj["context"]["trace_id"] = self._remove_0x_from_start(obj["context"]["trace_id"])
                obj["context"]["span_id"] = self._remove_0x_from_start(obj["context"]["span_id"])
            span_list["batch"].append(obj)
        
        # Store the spans for test assertions
        self.exported_spans.append(span_list)
        
        logger.debug(f"MockSpanExporter: Captured {len(spans)} spans")
        return SpanExportResult.SUCCESS
    
    def _validate_span_attributes(self, span: ReadableSpan) -> None:
        """Validate span attributes against configured checks."""
        for key, expected_value in self.attributes_to_check.items():
            try:
                # Skip app hosting attributes for non-workflow spans
                if key in ['entity.2.name', 'entity.2.type'] and span._attributes.get('span.type') != 'workflow':
                    continue
                    
                # Check if the attribute exists
                if key not in span._attributes:
                    error_msg = f"Expected attribute '{key}' not found in span '{span.name}'. Available attributes: {list(span._attributes.keys())}"
                    if self.strict_mode:
                        raise AssertionError(error_msg)
                    else:
                        logger.warning(error_msg)
                        continue
                
                actual_value = span._attributes[key]
                if actual_value != expected_value:
                    error_msg = f"Attribute mismatch in span '{span.name}' for key '{key}': expected '{expected_value}', got '{actual_value}'"
                    if self.strict_mode:
                        raise AssertionError(error_msg)
                    else:
                        logger.error(error_msg)
                        
            except Exception as e:
                error_msg = f"Error checking attribute '{key}' in span '{span.name}': {e}"
                if self.strict_mode:
                    logger.error(error_msg)
                    raise
                else:
                    logger.warning(error_msg)
    
    # Span data access methods
    def get_exported_spans(self) -> List[Dict[str, Any]]:
        """Get all exported spans for test assertions."""
        return self.exported_spans
    
    def get_latest_batch(self) -> Optional[Dict[str, Any]]:
        """Get the most recent batch of exported spans."""
        return self.exported_spans[-1] if self.exported_spans else None
    
    def clear_spans(self):
        """Clear all stored spans."""
        self.exported_spans.clear()
    
    # Attribute validation methods
    def set_trace_check(self, attributes_to_check: Dict[str, str]):
        """Set attributes to validate during span export."""
        self.attributes_to_check.update(attributes_to_check)
    
    def clear_trace_check(self):
        """Clear all attribute validation checks."""
        self.attributes_to_check.clear()
    
    # Standard exporter methods
    def shutdown(self) -> None:
        """Properly shutdown the exporter and clear all state."""
        if self._closed:
            logger.debug("Exporter already shutdown, ignoring call")
            return
        
        # Clear all state
        self.clear_trace_check()
        self.clear_spans()
        self.current_trace_id = None
        self.current_file_path = None
        self._closed = True
        logger.debug("MockSpanExporter shutdown completed")
    
    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush - no-op for mock exporter."""
        return True
    
    # Context manager support
    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures proper cleanup"""
        self.shutdown()
        return False
    
    # Utility methods
    def _remove_0x_from_start(self, my_str: str) -> str:
        """Remove the first occurrence of 0x from the string."""
        if my_str.startswith("0x"):
            return my_str.replace("0x", "", 1)
        return my_str