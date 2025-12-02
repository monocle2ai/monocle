import pytest
import os
from monocle_test_tools import (
    MonocleValidator, TestCase, TestSpan, Entity, DefaultComparer
)
from span_loader import JSONSpanLoader

@pytest.fixture(scope="module")
def setup():
    """Fixture to create a MonocleValidator instance for testing."""
    validator = MonocleValidator()
    current_script_path = os.path.abspath(__file__)
    spans = JSONSpanLoader.from_json(os.path.join(os.path.dirname(current_script_path), "traces/trace1.json"))
    validator.memory_exporter.export(spans)
    yield validator
    validator.memory_exporter.clear()

def test_validator_initialization(setup):
    """Test that MonocleValidator can be initialized."""
    assert isinstance(setup, MonocleValidator)

def test_tool_invocation_span(setup):
    """Test validation of a tool invocation span."""
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
    setup.validate(test_case)

def test_tool_invocation_span_negative(setup):
    """Test validation of a tool invocation span."""
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
    setup.validate(test_case)

def test_agent_invocation(setup):
    """Test validation of a tool invocation span."""
    test_case = TestCase(
        test_input=("Book a flight from San Francisco to Mumbai for 26th Nov 2025. Book a two queen room at Marriot Intercontinental at Juhu, Mumbai for 27th Nov 2025 for 4 nights.",),
        test_spans=[{
            "span_type": "agentic.invocation",
            "entities": [
                {"type": "agent", "name": "adk_hotel_booking_agent_5"}
            ]
        }]
    )

    setup.validate(test_case)

if __name__ == "__main__":
    pytest.main([__file__])