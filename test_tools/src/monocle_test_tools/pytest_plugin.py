import os
from datetime import datetime
import pytest
from .fluent_api import TraceAssertion
from .testcase import FluentTestCase
from . import eval_matrix


def pytest_addoption(parser) -> None:
    """Register the opt-in --monocle-eval-matrix option.

    See `eval_matrix.py` for the recorder implementation. Off by default:
    a plain `pytest` invocation with neither this flag nor the
    MONOCLE_EVAL_MATRIX env var set records nothing and writes no file.
    """
    eval_matrix.pytest_addoption(parser)


def pytest_sessionfinish(session) -> None:
    """If the eval-result-matrix recorder is enabled, write the collected
    records to the resolved output path."""
    eval_matrix.pytest_sessionfinish(session)


def pytest_make_parametrize_id(config, val, argname):
    """Name a parametrized test case after itself rather than its index.

    A suite built by ``setup_test_cases()`` is one test function over many cases,
    so the default ids (``testcase0``, ``testcase1``, ...) leave a failure
    saying nothing about *which* fact broke. Returning the case's own name makes
    the node id ``test_something[<fact id or case name>]``, which is also what
    you paste back to re-run just that case.

    This only supplies the bracketed part -- pytest prefixes the test function
    name itself.

    A ``FluentTestCase`` is named by its ``name`` whatever the argument is
    called, since the type is unambiguous. A plain dict is only named when the
    argument is called ``testcase``: without that gate, any unrelated test
    parametrizing over dicts with a ``"name"`` key would silently get new node
    ids, breaking anything that matches on them.

    Returns None for everything else, which leaves pytest's own id generation
    untouched.
    """
    if isinstance(val, FluentTestCase):
        name = val.name
    elif isinstance(val, dict) and argname == "testcase":
        name = val.get("name")
    else:
        return None
    return str(name) if name is not None else None

@pytest.fixture(scope="session", autouse=True)
def run_once_at_start_of_session():
    """
    This fixture runs once at the start of the pytest session.
    Place any setup code here that should execute only once before any tests run.
    """
    # Set LOCAL_RUN_ID only if not already set (to preserve it across session)
    if "LOCAL_RUN_ID" not in os.environ:
        os.environ["LOCAL_RUN_ID"] = datetime.now().isoformat()
    yield

@pytest.fixture()
def monocle_trace_asserter(request:pytest.FixtureRequest):
    """
    Provides a fresh TraceAssertion instance for each test.
    
    This fixture automatically handles cleanup and ensures test isolation.
    Each test gets its own clean asserter with cleared memory and empty spans.
    
    Example:
        def test_my_agent(monocle_trace_asserter):
            monocle_trace_asserter.run_agent(my_agent, "google_adk", "my_task")
            
            # Make assertions
            monocle_trace_asserter.called_tool("my_tool") \\
                .has_input("expected input") \\
                .contains_output("expected output")
    """
    traceAssertion = TraceAssertion.get_trace_asserter()
    token = traceAssertion.validator.pre_test_run_setup(request.node.name)
    exception_message = None
    try:
        result = yield traceAssertion
    except Exception as e:
        # Capture the actual exception message when test fails
        exception_message = str(e)
        raise
    finally:
        is_test_failed = _is_test_failed(request)
        if is_test_failed:
            # Priority: 1) Captured exception 2) pytest report exception 3) trace assertions
            if exception_message:
                assertion_messages = exception_message
            elif hasattr(request.node, 'rep_call') and hasattr(request.node.rep_call, 'longrepr'):
                # Try to get the actual exception from pytest's representation
                longrepr = request.node.rep_call.longrepr
                
                # If longrepr has reprcrash, use it (it has the exception message)
                if hasattr(longrepr, 'reprcrash') and longrepr.reprcrash:
                    assertion_messages = longrepr.reprcrash.message
                # If longrepr has reprtraceback with an exception entry, use that
                elif hasattr(longrepr, 'reprtraceback'):
                    # Get the exception info from the traceback
                    longrepr_str = str(longrepr)
                    # The exception message is usually after the last "E   " line
                    lines = longrepr_str.split('\n')
                    exc_lines = [line[4:] for line in lines if line.startswith('E   ')]
                    if exc_lines:
                        assertion_messages = ' '.join(exc_lines)
                    else:
                        # Fallback: look for ValueError, TypeError, etc.
                        for line in reversed(lines):
                            if 'Error:' in line or 'Exception:' in line:
                                assertion_messages = line.strip()
                                break
                        else:
                            assertion_messages = str(longrepr)
                else:
                    assertion_messages = str(longrepr)
            else:
                assertion_messages = traceAssertion.get_assertion_messages()
        else:
            assertion_messages = None
        traceAssertion.validator.post_test_cleanup(token, request.node.name, is_test_failed,
                                    assertion_messages, skip_export=traceAssertion._skip_export)

        # Opt-in eval-result-matrix recorder: self-skips when disabled, or
        # when this test never called check_eval (no `_last_eval` stash).
        eval_matrix.record_eval_row_for(request.config, request, traceAssertion)

        # Cleanup trace asserter (triggers eval cleanup including trace deletion)
        traceAssertion.cleanup()

@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Capture test reports and modify based on trace assertions."""
    outcome = yield
    rep = outcome.get_result()
    
    # Store report
    setattr(item, f"rep_{rep.when}", rep)
    
    # After test call phase, check trace assertions
    if rep.when == "call" and rep.outcome == "passed":
        traceAssertion:TraceAssertion = TraceAssertion()
        if traceAssertion.has_assertions():
            rep.outcome = "failed"

            rep.longrepr = traceAssertion.get_assertion_messages()
            _apply_xfail_to_deferred_failure(item, rep)


def _apply_xfail_to_deferred_failure(item, rep) -> None:
    """Honour @pytest.mark.xfail for assertions that were recorded, not raised.

    Fluent assertions record instead of raising, so pytest's own xfail
    hookwrapper already saw a passing call and left the marker unapplied.
    """
    if item.config.getoption("runxfail", False):
        return
    marker = item.get_closest_marker("xfail")
    if marker is None:
        return
    # A falsy literal condition disables it; leave string conditions to pytest.
    condition = marker.args[0] if marker.args else marker.kwargs.get("condition", True)
    if not isinstance(condition, str) and not condition:
        return
    # `raises=` cannot match a failure that never produced an exception.
    if marker.kwargs.get("raises") is not None:
        return
    rep.outcome = "skipped"
    rep.wasxfail = marker.kwargs.get("reason", "")

def _is_test_failed(request:pytest.FixtureRequest) -> bool:
    """Check if the test has failed based on the pytest request object."""
    return request.node.rep_call.passed == False if hasattr(request.node, "rep_call") else False

