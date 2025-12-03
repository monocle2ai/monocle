import pytest
from .fluent_api import TraceAssertion

@pytest.fixture()
def monocle_trace_asserter():
    """
    Provides a fresh TraceAssertion instance for each test.
    
    This fixture automatically handles cleanup and ensures test isolation.
    Each test gets its own clean asserter with cleared memory and empty spans.
    
    Example:
        def test_my_agent(monocle_trace_asserter):
            # Load your trace data
            monocle_trace_asserter.memory_exporter.export(spans)
            
            # Make assertions
            monocle_trace_asserter.called_tool("my_tool") \\
                .has_input("expected input") \\
                .contains_output("expected output")
    """
    return TraceAssertion.get_trace_asserter()