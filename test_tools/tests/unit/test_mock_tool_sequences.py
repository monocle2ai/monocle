"""A mock tool can answer each call differently: a list of responses in call order, or a handler
that computes the response from the tool's arguments and can keep state for the whole test case."""
import json
from contextlib import contextmanager

import pytest
from opentelemetry.context import attach, detach
from opentelemetry.sdk.trace import TracerProvider
from pydantic import ValidationError

from monocle_apptrace import stop_scope
from monocle_apptrace.instrumentation.common.span_handler import SpanHandler
from monocle_apptrace.instrumentation.common.utils import MonocleSpanException
from monocle_test_tools import MockTool, MonocleValidator

_tracer = TracerProvider().get_tracer("mock-tool-sequence-tests")


@pytest.fixture
def validator():
    validator = MonocleValidator()
    validator.reset_mock_tool_state()
    yield validator
    validator.reset_mock_tool_state()


@contextmanager
def agent_run(validator, mock_tools):
    """One agent run (one turn): the mocks are applied the way run_agent_async applies them."""
    token = attach(validator._set_wrapper_methods(mock_tools))

    def call(name, args, tool_type="tool.adk"):
        # What Monocle's wrapper does before a tool runs: open the tool span, record the
        # input, then ask whether to skip the real tool and what to return instead.
        with _tracer.start_as_current_span(name) as span:
            span.set_attribute("span.type", "agentic.tool.invocation")
            span.set_attribute("entity.1.name", name)
            span.set_attribute("entity.1.type", tool_type)
            span.add_event("data.input", {"input": json.dumps(args)})
            skipped, response = SpanHandler.skip_execution(span)
        assert skipped, f"{name} was not mocked"
        return response

    try:
        yield call
    finally:
        detach(token)


def lookup_flights(args, state):
    """A handler named by module path in a JSON scenario file."""
    return {"flights": [f"{args['origin']}-{args['destination']}"]}


# --- the model -------------------------------------------------------------------------------

def test_responses_alone_make_a_valid_mock():
    mock = MockTool(name="search_flights", type="tool.adk", responses=[[], ["FL101"]])
    assert mock.responses == [[], ["FL101"]]


def test_a_handler_alone_makes_a_valid_mock():
    mock = MockTool(name="search_flights", type="tool.adk", handler=lookup_flights)
    assert mock.handler is lookup_flights


def test_a_handler_can_be_given_as_a_module_path_in_json():
    mock = MockTool.model_validate({"name": "search_flights", "type": "tool.adk",
                                    "handler": f"{__name__}:lookup_flights"})
    assert mock.handler == f"{__name__}:lookup_flights"


@pytest.mark.parametrize("fields", [
    {},
    {"response": "a", "responses": ["b"]},
    {"response": "a", "handler": lookup_flights},
    {"responses": ["b"], "handler": lookup_flights},
    {"responses": []},
])
def test_a_mock_needs_exactly_one_way_to_respond(fields):
    with pytest.raises(ValidationError):
        MockTool(name="search_flights", type="tool.adk", **fields)


def test_on_exhausted_accepts_only_repeat_last_or_error():
    with pytest.raises(ValidationError):
        MockTool(name="t", type="tool.adk", responses=["a"], on_exhausted="cycle")


# --- responses in call order -------------------------------------------------------------------

def test_responses_are_returned_in_call_order(validator):
    mock = MockTool(name="search_flights", type="tool.adk", responses=[[], [{"flight_id": "FL101"}]])
    with agent_run(validator, [mock]) as call:
        assert call("search_flights", {"date": "2026-11-13"}) == []
        assert call("search_flights", {"date": "2026-11-14"}) == [{"flight_id": "FL101"}]


def test_the_last_response_repeats_when_the_list_runs_out(validator):
    mock = MockTool(name="get_status", type="tool.adk", responses=["pending", "shipped"])
    with agent_run(validator, [mock]) as call:
        assert [call("get_status", {}) for _ in range(3)] == ["pending", "shipped", "shipped"]


def test_running_out_of_responses_fails_the_call_when_asked(validator):
    mock = MockTool(name="get_status", type="tool.adk", responses=["pending"], on_exhausted="error")
    with agent_run(validator, [mock]) as call:
        assert call("get_status", {}) == "pending"
        with pytest.raises(MonocleSpanException, match="get_status.*2"):
            call("get_status", {})


def test_a_response_item_can_simulate_an_error_for_that_call(validator):
    mock = MockTool(name="book_flight", type="tool.adk",
                    responses=[{"raise_error": True, "error_message": "gateway timeout"}, {"status": "confirmed"}])
    with agent_run(validator, [mock]) as call:
        with pytest.raises(MonocleSpanException, match="gateway timeout"):
            call("book_flight", {"flight_id": "FL101"})
        assert call("book_flight", {"flight_id": "FL101"}) == {"status": "confirmed"}


def test_placeholders_are_filled_in_each_response(validator):
    mock = MockTool(name="book_flight", type="tool.adk", responses=["Booked {{flight_id}}"])
    with agent_run(validator, [mock]) as call:
        assert call("book_flight", {"flight_id": "FL101"}) == "Booked FL101"


def test_each_tool_counts_its_own_calls(validator):
    search = MockTool(name="search_flights", type="tool.adk", responses=["s1", "s2"])
    book = MockTool(name="book_flight", type="tool.adk", responses=["b1", "b2"])
    with agent_run(validator, [search, book]) as call:
        assert [call("search_flights", {}), call("book_flight", {}), call("search_flights", {})] == ["s1", "b1", "s2"]


# --- handlers ----------------------------------------------------------------------------------

def test_a_handler_computes_the_response_from_the_tool_arguments(validator):
    mock = MockTool(name="search_flights", type="tool.adk", handler=lookup_flights)
    with agent_run(validator, [mock]) as call:
        assert call("search_flights", {"origin": "BLR", "destination": "BOM"}) == {"flights": ["BLR-BOM"]}


def test_a_handler_named_by_module_path_is_resolved_at_call_time(validator):
    mock = MockTool.model_validate({"name": "search_flights", "type": "tool.adk",
                                    "handler": f"{__name__}:lookup_flights"})
    with agent_run(validator, [mock]) as call:
        assert call("search_flights", {"origin": "SFO", "destination": "JFK"}) == {"flights": ["SFO-JFK"]}


def test_handlers_share_state_within_a_test_case(validator):
    def search(args, state):
        state["last_search"] = ["FL101", "FL102"]
        return state["last_search"]

    def book(args, state):
        return {"status": "confirmed" if args["flight_id"] in state.get("last_search", []) else "unknown flight"}

    mocks = [MockTool(name="search_flights", type="tool.adk", handler=search),
             MockTool(name="book_flight", type="tool.adk", handler=book)]
    with agent_run(validator, mocks) as call:
        call("search_flights", {})
        assert call("book_flight", {"flight_id": "FL102"}) == {"status": "confirmed"}
        assert call("book_flight", {"flight_id": "FL999"}) == {"status": "unknown flight"}


# --- lifetime: one test case, across turns -----------------------------------------------------

def test_call_counts_continue_across_the_turns_of_one_test_case(validator):
    mock = MockTool(name="get_status", type="tool.adk", responses=["pending", "shipped"])
    with agent_run(validator, [mock]) as call:
        assert call("get_status", {}) == "pending"
    with agent_run(validator, [mock]) as call:     # the next turn applies the mocks again
        assert call("get_status", {}) == "shipped"


def test_handler_state_continues_across_the_turns_of_one_test_case(validator):
    def remember(args, state):
        state.setdefault("seen", []).append(args["q"])
        return list(state["seen"])

    mock = MockTool(name="search", type="tool.adk", handler=remember)
    with agent_run(validator, [mock]) as call:
        call("search", {"q": "a"})
    with agent_run(validator, [mock]) as call:
        assert call("search", {"q": "b"}) == ["a", "b"]


def test_a_new_test_case_starts_again_from_the_first_response(validator):
    mock = MockTool(name="get_status", type="tool.adk", responses=["pending", "shipped"])
    with agent_run(validator, [mock]) as call:
        assert call("get_status", {}) == "pending"
    stop_scope(validator.pre_test_run_setup("next_test_case"))
    with agent_run(validator, [mock]) as call:
        assert call("get_status", {}) == "pending"


# --- unchanged behaviour -------------------------------------------------------------------------

def test_a_single_response_mock_returns_the_same_response_every_call(validator):
    mock = MockTool(name="get_status", type="tool.adk", response="pending")
    with agent_run(validator, [mock]) as call:
        assert [call("get_status", {}) for _ in range(2)] == ["pending", "pending"]


def test_raise_error_still_fails_every_call_of_a_single_response_mock(validator):
    mock = MockTool(name="get_status", type="tool.adk", response=None, raise_error=True, error_message="down")
    with agent_run(validator, [mock]) as call:
        with pytest.raises(MonocleSpanException, match="down"):
            call("get_status", {})


def test_a_varying_mock_on_an_older_apptrace_says_to_upgrade_it(validator, monkeypatch):
    from monocle_test_tools import validator as validator_module
    monkeypatch.setattr(validator_module, "SKIP_EXECUTION_RESPONSE_FN", None)
    mock = MockTool(name="get_status", type="tool.adk", responses=["pending"])
    with pytest.raises(RuntimeError, match="pip install -U monocle_apptrace"):
        validator._set_wrapper_methods([mock])


@pytest.mark.parametrize("fields", [{"responses": ["a"]}, {"handler": lookup_flights}])
def test_raise_error_is_rejected_where_it_would_be_ignored(fields):
    # A varying mock fails single calls with an error item or by raising in the handler.
    with pytest.raises(ValidationError, match="raise_error"):
        MockTool(name="t", type="tool.adk", raise_error=True, **fields)
