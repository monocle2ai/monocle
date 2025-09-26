import pytest
import os
from monocle_test_tools import (
    MonocleValidator, TestCase, TestSpan, Entity, DefaultComparer
)
from span_loader import JSONSpanLoader

validator = MonocleValidator()

def test_validator_initialization():
    """Test that MonocleValidator can be initialized."""
    assert isinstance(validator, MonocleValidator)

def test_tool_invocation_span():
    """Test validation of a tool invocation span."""
    cwd1 = os.getcwd()
    spans = JSONSpanLoader.from_json("./test_tools/tests/unit/traces/trace1.json")
    validator.memory_exporter.export(spans)

    test_case = TestCase(
        test_spans=[
            TestSpan(span_type="agentic.tool.invocation", 
                entities=[
                    Entity(type="tool", name="adk_book_hotel_5"),
                    Entity(type="agent", name="adk_hotel_booking_agent_5")
                ],
                input = "{'city': 'Mumbai', 'hotel_name': 'Marriot Intercontinental'}",
                output= "{'status': 'success', 'message': 'Successfully booked a stay at Marriot Intercontinental in Mumbai.'}",
                comparer = DefaultComparer(),
            )
        ]
    )
    validator.validate(test_case)

def test_tool_invocation_span_negative():
    """Test validation of a tool invocation span."""
    cwd1 = os.getcwd()
    spans = JSONSpanLoader.from_json("./test_tools/tests/unit/traces/trace1.json")
    validator.memory_exporter.export(spans)

    test_case = TestCase(
        test_input = ("Book a flight from San Francisco to Mumbai for 26th Nov 2025. Book a two queen room at Marriot Intercontinental at Juhu, Mumbai for 27th Nov 2025 for 4 nights.",),
        test_spans=[
            TestSpan(span_type="agentic.tool.invocation", 
                entities=[
                    Entity(type="tool", name="weather_tool"),
                    Entity(type="agent", name="adk_hotel_booking_agent_5")
                ],
                test_type="negative",
            ),
        ]
    )
    validator.validate(test_case)

def test_agent_invocation():
    """Test validation of a tool invocation span."""
    cwd1 = os.getcwd()
    spans = JSONSpanLoader.from_json("./test_tools/tests/unit/traces/trace1.json")
    validator.memory_exporter.export(spans)

    test_case = TestCase(
        test_input=("Book a flight from San Francisco to Mumbai for 26th Nov 2025. Book a two queen room at Marriot Intercontinental at Juhu, Mumbai for 27th Nov 2025 for 4 nights.",),
        test_spans=[{
            "span_type": "agentic.invocation",
            "entities": [
                {"type": "agent", "name": "adk_hotel_booking_agent_5"}
            ]
        }]
    )

    validator.validate(test_case)
if __name__ == "__main__":
    pytest.main([__file__])