import pytest
from .fluent_api import TraceAssertion
from monocle_apptrace.instrumentation.common.scope_wrapper import start_scopes, stop_scope
from .constants import TEST_SCOPE_NAME
from .gitutils import get_git_context

@pytest.fixture
def monocle_trace_asserter(request):
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
    traceAssertion = TraceAssertion.get_trace_asserter()
    test_scope = {TEST_SCOPE_NAME: request.function.__name__}
    git_scopes = get_git_context()
    all_scopes = {**test_scope, **git_scopes}
    token = start_scopes(all_scopes)
    try:
        yield traceAssertion
    finally:
        traceAssertion.cleanup()
        stop_scope(token)
