import pytest
import os
from monocle_test_tools import TraceAssertion
from span_loader import JSONSpanLoader
os.environ["MONOCLE_EXPORT_FAILED_TESTS_ONLY"] = "true"

def test_tool_invocation_span(monocle_trace_asserter:TraceAssertion):
    monocle_trace_asserter.load_spans(JSONSpanLoader.load_spans("traces/trace1.json"))
    monocle_trace_asserter.called_tool("adk_book_hotel_5", "adk_hotel_booking_agent_5") \
        .has_input("{'city': 'Mumbai', 'hotel_name': 'Marriot Intercontinental'}") \
        .has_output("{'status': 'success', 'message': 'Successfully booked a stay at Marriot Intercontinental in Mumbai.'}") \
        .contains_input("Mumbai") \
        .contains_output("Successfully booked") \
        .does_not_contain_input("Delhi") \
        .does_not_contain_output("failed")

def test_agent_invocation(monocle_trace_asserter:TraceAssertion):
    monocle_trace_asserter.load_spans(JSONSpanLoader.load_spans("traces/trace1.json"))
    monocle_trace_asserter.called_agent("adk_hotel_booking_agent_5") \
        .has_input("Book a flight from San Francisco to Mumbai for 26th Nov 2025. Book a two queen room at Marriot Intercontinental at Juhu, Mumbai for 27th Nov 2025 for 4 nights.") \
        .contains_output("I have booked a stay at Marriot Intercontinental in Mumbai.") \
        .does_not_have_output("cancel the booking") \
        .does_not_have_output("failed")

def test_span_attribute_assertions(monocle_trace_asserter:TraceAssertion):
    trace_path = os.path.join(os.path.dirname(__file__), "traces/trace1.json")
    monocle_trace_asserter.with_trace_source(source="file", trace_path=trace_path)
    monocle_trace_asserter.called_tool("adk_book_hotel_5", "adk_hotel_booking_agent_5") \
        .has_attribute("entity.1.type", "tool.adk") \
        .has_attribute("workflow.name") \
        .does_not_have_attribute("entity.1.type", "tool.openai")

def test_has_attribute_with_chaining(monocle_trace_asserter:TraceAssertion):
    trace_path = os.path.join(os.path.dirname(__file__), "traces/trace1.json")
    loaded = monocle_trace_asserter.with_trace_source(source="file", trace_path=trace_path)

    # Named parameters
    matching = loaded.has_attribute(attribute_name="span.type", expected="inference") \
        .has_event(event_name="metadata", attribute_name="total_tokens", expected=229)
    assert matching._filtered_spans is not None
    assert len(matching._filtered_spans) == 1
    assert not matching.has_assertions()

    # Positional parameters
    positional = loaded.has_attribute("workflow.name")
    assert not positional.has_assertions()

def test_where_generic_selector(monocle_trace_asserter:TraceAssertion):
    trace_path = os.path.join(os.path.dirname(__file__), "traces/trace1.json")
    loaded = monocle_trace_asserter.with_trace_source(source="file", trace_path=trace_path)

    # Combine an attribute match and an event match on the same span
    matching = loaded.where(
        attribute={"span.type": "inference"},
        event={"name": "metadata", "attributes": {"total_tokens": 229}},
    )
    assert matching._filtered_spans is not None
    assert len(matching._filtered_spans) == 1
    assert not matching.has_assertions()

    # Predicate-based selection
    pred = loaded.where(predicate=lambda span: span.attributes.get("entity.count") == 2)
    assert pred._filtered_spans
    assert not pred.has_assertions()

def test_where_filter_engine_matches_and_misses(monocle_trace_asserter:TraceAssertion):
    trace_path = os.path.join(os.path.dirname(__file__), "traces/trace1.json")
    loaded = monocle_trace_asserter.with_trace_source(source="file", trace_path=trace_path)
    spans = loaded.validator.spans
    assert spans is not None

    # AND across attribute + event on the same span
    assert loaded._filter_spans_where(
        spans, {"span.type": "inference"}, {"name": "metadata", "attributes": {"total_tokens": 229}}, None
    )
    # Non-matching attribute yields no spans
    assert not loaded._filter_spans_where(spans, {"span.type": "does.not.exist"}, None, None)
    # Predicate-based filtering
    assert loaded._filter_spans_where(spans, None, None, lambda s: s.attributes.get("entity.count") == 2)
    # Conflicting criteria across facets: attribute present but event absent -> no match
    assert not loaded._filter_spans_where(spans, {"span.type": "inference"}, {"name": "nonexistent.event"}, None)

def test_where_requires_a_criterion(monocle_trace_asserter:TraceAssertion):
    trace_path = os.path.join(os.path.dirname(__file__), "traces/trace1.json")
    loaded = monocle_trace_asserter.with_trace_source(source="file", trace_path=trace_path)
    with pytest.raises(ValueError):
        loaded.where()
    with pytest.raises(ValueError):
        loaded.does_not_match()

def test_does_not_match_generic_selector(monocle_trace_asserter:TraceAssertion):
    trace_path = os.path.join(os.path.dirname(__file__), "traces/trace1.json")
    loaded = monocle_trace_asserter.with_trace_source(source="file", trace_path=trace_path)

    # No span carries this attribute, so does_not_match passes (no recorded assertion)
    ok = loaded.does_not_match(attribute={"span.type": "does.not.exist"})
    assert not ok.has_assertions()

def test_generic_span_event_assertions(monocle_trace_asserter:TraceAssertion):
    trace_path = os.path.join(os.path.dirname(__file__), "traces/trace1.json")
    loaded = monocle_trace_asserter.with_trace_source(source="file", trace_path=trace_path)

    matching = loaded.has_event(
        event_name="metadata", attribute_name="total_tokens", expected=229
    ).has_attribute("span.type", "inference")

    assert matching._filtered_spans is not None
    assert len(matching._filtered_spans) == 1

def test_event_filter_distinguishes_missing_and_non_matching_values(monocle_trace_asserter:TraceAssertion):
    trace_path = os.path.join(os.path.dirname(__file__), "traces/trace1.json")
    loaded = monocle_trace_asserter.with_trace_source(source="file", trace_path=trace_path)
    spans = loaded.validator.spans

    assert spans is not None
    assert loaded._filter_spans_by_event(spans, "metadata", "total_tokens", 229)
    assert not loaded._filter_spans_by_event(spans, "metadata", "missing", None)
    assert not loaded._filter_spans_by_event(spans, "metadata", "total_tokens", "229")

if __name__ == "__main__":
    pytest.main([__file__])
