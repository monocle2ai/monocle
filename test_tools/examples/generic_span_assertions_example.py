"""
Example demonstrating generic span attribute and event assertions.

This shows how to use the generic API to validate any span attribute or event,
which is useful for:
1. Validating Monocle instrumentation itself
2. Testing custom/ad-hoc instrumentation
3. Asserting any OpenTelemetry span property
"""

from monocle_test_tools import TraceAssertion


def example_basic_attribute_assertions():
    """Basic attribute assertions using has_attribute()"""
    asserter = TraceAssertion.get_trace_asserter()
    
    # Load test trace from file
    asserter.with_trace_source(source="file", trace_path="traces/trace1.json")
    
    # Assert a span has a specific attribute with expected value
    asserter.has_attribute(attribute_name="span.type", expected="inference")
    
    # Assert a span has an attribute (presence check only)
    asserter.has_attribute(attribute_name="workflow.name")
    
    # Negative assertion - no span should have this attribute/value
    asserter.does_not_have_attribute(attribute_name="entity.1.type", expected="tool.openai")
    
    print("✓ Basic attribute assertions passed")


def example_event_assertions():
    """Event assertions using has_event()"""
    asserter = TraceAssertion.get_trace_asserter()
    asserter.with_trace_source(source="file", trace_path="traces/trace1.json")
    
    # Assert a span has an event with a specific attribute value
    asserter.has_event(
        event_name="metadata",
        attribute_name="total_tokens",
        expected=229
    )
    
    # Assert a span has an event (presence check only)
    asserter.has_event(event_name="metadata")
    
    print("✓ Event assertions passed")


def example_chaining_assertions():
    """Chain multiple assertions to narrow down spans"""
    asserter = TraceAssertion.get_trace_asserter()
    asserter.with_trace_source(source="file", trace_path="traces/trace1.json")
    
    # Chain attribute and event assertions
    # This filters to spans that match ALL criteria
    asserter \
        .has_attribute(attribute_name="span.type", expected="inference") \
        .has_event(event_name="metadata", attribute_name="total_tokens", expected=229)
    
    print("✓ Chained assertions passed")


def example_where_generic_selector():
    """Use where() for complex multi-criteria filtering"""
    asserter = TraceAssertion.get_trace_asserter()
    asserter.with_trace_source(source="file", trace_path="traces/trace1.json")
    
    # Combine attribute and event criteria in one call
    asserter.where(
        attribute={"span.type": "inference"},
        event={"name": "metadata", "attributes": {"total_tokens": 229}}
    )
    
    # Use multiple attributes
    asserter.where(
        attribute={
            "span.type": "inference",
            "workflow.name": None  # None = presence check
        }
    )
    
    # Use a custom predicate for complex logic
    asserter.where(
        predicate=lambda span: span.attributes.get("entity.count", 0) >= 2
    )
    
    # Negative assertion - assert no span matches
    asserter.does_not_match(
        attribute={"span.type": "does.not.exist"}
    )
    
    print("✓ Generic where() selector passed")


def example_validate_monocle_instrumentation():
    """
    Real-world example: Validate Monocle's instrumentation of a tool call
    """
    asserter = TraceAssertion.get_trace_asserter()
    asserter.with_trace_source(source="file", trace_path="traces/trace1.json")
    
    # Validate tool invocation has correct Monocle attributes
    asserter \
        .called_tool("adk_book_hotel_5", "adk_hotel_booking_agent_5") \
        .has_attribute(attribute_name="entity.1.type", expected="tool.adk") \
        .has_attribute(attribute_name="entity.1.name", expected="adk_book_hotel_5") \
        .has_attribute(attribute_name="span.type", expected="tool.adk") \
        .has_attribute(attribute_name="workflow.name")  # Must have workflow context
    
    print("✓ Monocle instrumentation validation passed")


def example_validate_agentic_turn():
    """
    Validate an agentic turn has correct metadata
    """
    asserter = TraceAssertion.get_trace_asserter()
    asserter.with_trace_source(source="file", trace_path="traces/trace1.json")
    
    # Find all agentic turns with token metadata
    asserter.where(
        attribute={"span.type": "inference"},
        event={"name": "metadata", "attributes": {
            "total_tokens": None,  # Must have total_tokens (any value)
        }}
    )
    
    print("✓ Agentic turn validation passed")


if __name__ == "__main__":
    
    example_basic_attribute_assertions()
    example_event_assertions()
    example_chaining_assertions()
    example_where_generic_selector()
    example_validate_monocle_instrumentation()
    example_validate_agentic_turn()
    
