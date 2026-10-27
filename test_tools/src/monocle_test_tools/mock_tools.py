"""Mock tools that answer each call differently.

A ``MockTool`` gives one fixed ``response`` for every call, a list of ``responses`` in call order,
or a ``handler`` that computes the response from the tool's arguments. Call counts and the handler
state live in a ``MockToolState`` that lasts for one test case, across all of its turns, so a
multi-turn conversation sees the second response on its second call even when that call is in a
later turn.
"""
import importlib
from dataclasses import dataclass, field
from typing import Any, Callable, Union

from monocle_apptrace.instrumentation.common.utils import MonocleSpanException

ON_EXHAUSTED_REPEAT_LAST = "repeat_last"
ON_EXHAUSTED_ERROR = "error"

# A response item that simulates an error for that call instead of returning a value.
_ERROR_ITEM_KEYS = {"raise_error", "error_message"}

Handler = Callable[[dict, dict], Any]


@dataclass
class MockToolState:
    """Per test case: how many times each mocked tool was called, and the handlers' shared state."""

    calls: dict[str, int] = field(default_factory=dict)
    data: dict = field(default_factory=dict)

    def next_call(self, tool_name: str) -> int:
        """Count a call of ``tool_name`` and return its 0-based index in this test case."""
        index = self.calls.get(tool_name, 0)
        self.calls[tool_name] = index + 1
        return index


def resolve_handler(handler: Union[str, Handler]) -> Handler:
    """A handler given as ``"package.module:function"`` (from a JSON or YAML file), or a callable."""
    if callable(handler):
        return handler
    module_name, _, function_name = handler.partition(":")
    if not module_name or not function_name:
        raise ValueError(f"handler {handler!r} must be a callable or 'package.module:function'")
    function = getattr(importlib.import_module(module_name), function_name, None)
    if not callable(function):
        raise ValueError(f"handler {handler!r} does not name a callable")
    return function


def is_error_item(item: Any) -> bool:
    """``{"raise_error": true, "error_message": "..."}`` makes that call fail."""
    return isinstance(item, dict) and item.get("raise_error") is True and set(item) <= _ERROR_ITEM_KEYS


def build_response_fn(name: str, responses: Union[list, None], handler: Union[str, Handler, None],
                      on_exhausted: str, state: MockToolState) -> Callable[[dict], Any]:
    """The per-call response function that apptrace calls with the tool's arguments."""
    def respond(args: dict) -> Any:
        index = state.next_call(name)
        if handler is not None:
            return resolve_handler(handler)(args, state.data)
        if index < len(responses):
            item = responses[index]
        elif on_exhausted == ON_EXHAUSTED_ERROR:
            raise MonocleSpanException(
                f"MockTool {name}: no response left for call {index + 1} ({len(responses)} given)")
        else:
            item = responses[-1]
        if is_error_item(item):
            raise MonocleSpanException(item.get("error_message", "Simulated tool error"))
        return item

    return respond
