"""
In-memory span exporter for testing SpanFilter / FilteredSpanExporter pipelines.

Usage with FilteredSpanExporter
--------------------------------
    from monocle_apptrace.exporters.span_filter import SpanFilter, FilteredSpanExporter
    from tests.common.span_filter_test_exporter import SpanFilterTestExporter

    test_exporter = SpanFilterTestExporter()
    filtered_exporter = FilteredSpanExporter(
        base_exporter=test_exporter,
        span_filter=SpanFilter({
            "span_types_to_include": ["inference"],
            "fields_to_include": {
                "attributes": ["entity.1.type", "scope.*"],
                "events": [{"name": "metadata"}],
            },
        }),
    )

    filtered_exporter.export(spans)

    # Assertions
    assert test_exporter.export_count == 1
    attrs = test_exporter.get_span_attributes(0)
    assert "entity.1.type" in attrs
    assert test_exporter.find_spans_by_type("inference")

Usage standalone (no FilteredSpanExporter)
-------------------------------------------
    test_exporter = SpanFilterTestExporter(span_filter=SpanFilter(config))
    test_exporter.export(spans)
    ...
"""

import json
from typing import Any, Dict, List, Optional, Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExportResult

from monocle_apptrace.exporters.span_filter import SpanFilter


class SpanFilterTestExporter:
    """Concrete in-memory exporter that captures filtered span data for assertions.

    Can be used in two ways:

    1. As the ``base_exporter`` inside ``FilteredSpanExporter`` — the exporter
       receives already-filtered ``FilteredReadableSpan`` objects and stores their
       JSON representation.

    2. Standalone with an optional ``span_filter`` — applies the filter itself
       before capturing, so no ``FilteredSpanExporter`` wrapper is needed.
    """

    def __init__(self, span_filter: Optional[SpanFilter] = None) -> None:
        self._span_filter = span_filter
        self._exported: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # SpanExporter interface
    # ------------------------------------------------------------------

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        for span in spans:
            if self._span_filter is not None:
                data = self._span_filter.filter(span)
            else:
                # Spans arriving here may already be FilteredReadableSpan objects
                # whose to_json() returns filtered data.
                try:
                    data = json.loads(span.to_json())
                except Exception:
                    data = None

            if data is not None:
                self._exported.append(data)

        return SpanExportResult.SUCCESS

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True

    def shutdown(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Test helpers
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all captured spans between test cases."""
        self._exported.clear()

    @property
    def export_count(self) -> int:
        """Number of spans that passed the filter and were exported."""
        return len(self._exported)

    @property
    def exported_spans(self) -> List[Dict[str, Any]]:
        """All captured span dicts in export order."""
        return list(self._exported)

    def get_span(self, index: int) -> Dict[str, Any]:
        """Return the captured span dict at *index*."""
        return self._exported[index]

    def get_span_attributes(self, index: int) -> Dict[str, Any]:
        """Return the attributes dict of the span at *index*."""
        return self._exported[index].get("attributes", {})

    def get_span_events(self, index: int) -> List[Dict[str, Any]]:
        """Return the events list of the span at *index*."""
        return self._exported[index].get("events", [])

    def get_event_attributes(self, span_index: int, event_name: str) -> Dict[str, Any]:
        """Return attributes of the first event matching *event_name* in a span."""
        for event in self.get_span_events(span_index):
            if event.get("name") == event_name:
                return event.get("attributes", {})
        return {}

    def find_spans_by_type(self, span_type: str) -> List[Dict[str, Any]]:
        """Return all captured spans whose ``span.type`` attribute matches *span_type*."""
        return [
            s for s in self._exported
            if s.get("attributes", {}).get("span.type") == span_type
        ]

    def find_spans_by_name(self, name: str) -> List[Dict[str, Any]]:
        """Return all captured spans whose ``name`` field matches *name*."""
        return [s for s in self._exported if s.get("name") == name]

    def assert_exported_count(self, expected: int) -> None:
        """Raise AssertionError if export count does not equal *expected*."""
        actual = self.export_count
        assert actual == expected, (
            f"Expected {expected} exported span(s), got {actual}"
        )

    def assert_attribute_present(self, span_index: int, attribute_key: str) -> None:
        """Raise AssertionError if *attribute_key* is missing from a span's attributes."""
        attrs = self.get_span_attributes(span_index)
        assert attribute_key in attrs, (
            f"Attribute '{attribute_key}' not found in span[{span_index}]. "
            f"Present keys: {sorted(attrs)}"
        )

    def assert_attribute_absent(self, span_index: int, attribute_key: str) -> None:
        """Raise AssertionError if *attribute_key* is present in a span's attributes."""
        attrs = self.get_span_attributes(span_index)
        assert attribute_key not in attrs, (
            f"Attribute '{attribute_key}' should be absent from span[{span_index}] "
            f"but was found with value: {attrs[attribute_key]!r}"
        )

    def assert_event_present(self, span_index: int, event_name: str) -> None:
        """Raise AssertionError if no event named *event_name* exists in a span."""
        names = [e.get("name") for e in self.get_span_events(span_index)]
        assert event_name in names, (
            f"Event '{event_name}' not found in span[{span_index}]. "
            f"Present events: {names}"
        )

    def assert_event_absent(self, span_index: int, event_name: str) -> None:
        """Raise AssertionError if an event named *event_name* is present in a span."""
        names = [e.get("name") for e in self.get_span_events(span_index)]
        assert event_name not in names, (
            f"Event '{event_name}' should be absent from span[{span_index}] "
            f"but was found."
        )
