"""Stopping a model call that one of the guardrails in effect triggers on."""
import dataclasses
import inspect
import logging
from collections import OrderedDict
from collections.abc import Mapping
from typing import Callable, Iterable, Iterator, List, Optional

from opentelemetry.trace import get_current_span

from monocle_apptrace.guardrails.guardrail import INPUT, OUTPUT, Guardrail, Trigger, guardrails_in_effect, watching_only
from monocle_apptrace.instrumentation.common.constants import (
    GUARDRAIL_ERROR_CODE, GUARDRAIL_NAME, GUARDRAIL_REASON, GUARDRAIL_SIDE, GUARDRAIL_WATCHED, SPAN_TYPE_KEY,
    SPAN_TYPES,
)
from monocle_apptrace.instrumentation.common.utils import get_current_monocle_span, model_call_inference

logger = logging.getLogger(__name__)

STOPPED_TRACES_REMEMBERED = 1000
DEEPEST_NESTING_CHECKED = 20
stopped_traces: "OrderedDict[int, Trigger]" = OrderedDict()


class GuardrailTriggered(Exception):
    """Raised in place of a model call, or of its response, that a guardrail stopped."""
    code = GUARDRAIL_ERROR_CODE

    def __init__(self, trigger: Trigger):
        super().__init__(f"Kill switch triggered by guardrail '{trigger.guardrail}' on the {trigger.side} of a model call.")
        self.trigger = trigger


class GuardedCall:
    """A model method that checks its input and its response against the guardrails in effect."""

    def __init__(self, model_method: Callable, inference: dict, guardrails: List[Guardrail], watch_only: bool):
        self.model_method = model_method
        self.inference = inference
        self.guardrails = guardrails
        self.watch_only = watch_only

    def __call__(self, *args, **kwargs):
        refuse_a_stopped_trace()
        self.check(INPUT, self.inference.get("guarded_input", whole_call), args, kwargs)
        response = self.model_method(*args, **kwargs)
        stream_was_asked_for = self.asks_for_a_stream(kwargs)
        if inspect.isawaitable(response):
            return self.checked_once_awaited(response, stream_was_asked_for)
        return self.checked(response, stream_was_asked_for)

    async def checked_once_awaited(self, pending_response, stream_was_asked_for: bool):
        return self.checked(await pending_response, stream_was_asked_for)

    def checked(self, response, stream_was_asked_for: bool):
        """The response, once no guardrail triggers on it. A streamed response is returned unchecked."""
        if stream_was_asked_for or inspect.isgenerator(response) or inspect.isasyncgen(response):
            return response
        self.check(OUTPUT, self.inference.get("guarded_response", response_as_returned), response)
        return response

    def asks_for_a_stream(self, kwargs: dict) -> bool:
        is_auto_close = self.inference.get("is_auto_close")
        return is_auto_close is not None and not is_auto_close(kwargs)

    def check(self, side: str, read: Callable, *read_from) -> None:
        """Stop the call if a guardrail triggers on this side of it. ``read`` gives what this side holds."""
        guardrails_for_this_side = [guardrail for guardrail in self.guardrails if side in guardrail.sides]
        if not guardrails_for_this_side:
            return
        trigger = first_trigger(guardrails_for_this_side, side, text_held_in(side, read, *read_from))
        if trigger is None:
            return
        if self.watch_only:
            record_trigger(get_current_monocle_span(), trigger, watched=True)
            return
        remember_stopped_trace(trigger)
        raise GuardrailTriggered(trigger)


def guarded_call(to_wrap: dict, model_method: Callable) -> Callable:
    """The method to call at this wrap site: one that checks the guardrails in effect, or the method itself."""
    inference = model_call_inference(to_wrap)
    if inference is None:
        return model_method
    try:
        guardrails = guardrails_in_effect()
    except Exception as e:
        logger.warning("No guardrail checked the model call: %s", e)
        return model_method
    if not guardrails:
        return model_method
    return GuardedCall(model_method, inference, guardrails, watching_only())


def record_triggered_guardrail(span, error: Optional[BaseException]) -> None:
    """Mark an inference span whose call failed because a guardrail stopped it, or stopped a model call beneath it."""
    trigger = trigger_that_caused(error)
    if trigger is not None:
        record_trigger(span, trigger, watched=False)


def record_trigger(span, trigger: Trigger, watched: bool) -> None:
    if not is_inference(span):
        return
    span.set_attribute(GUARDRAIL_NAME, trigger.guardrail)
    span.set_attribute(GUARDRAIL_SIDE, trigger.side)
    span.set_attribute(GUARDRAIL_REASON, trigger.reason)
    if watched:
        span.set_attribute(GUARDRAIL_WATCHED, True)


def is_inference(span) -> bool:
    if not span.is_recording():
        return False
    return str(span.attributes.get(SPAN_TYPE_KEY, "")).startswith(SPAN_TYPES.INFERENCE)


def trigger_that_caused(error: Optional[BaseException]) -> Optional[Trigger]:
    """The trigger behind this error, also when a library has wrapped a stopped call's error in its own."""
    already_looked_at = set()
    while error is not None and id(error) not in already_looked_at:
        if isinstance(error, GuardrailTriggered):
            return error.trigger
        already_looked_at.add(id(error))
        error = error.__cause__ or error.__context__
    return None


def first_trigger(guardrails: List[Guardrail], side: str, text: str) -> Optional[Trigger]:
    for guardrail in guardrails:
        trigger = guardrail.trigger_on(side, text)
        if trigger is not None:
            return trigger
    return None


def refuse_a_stopped_trace() -> None:
    trigger = stopped_traces.get(current_trace())
    if trigger is not None:
        raise GuardrailTriggered(trigger)


def remember_stopped_trace(trigger: Trigger) -> None:
    """Every later model call in the trace this one belongs to is refused."""
    trace = current_trace()
    if not trace:
        return
    stopped_traces[trace] = trigger
    if len(stopped_traces) > STOPPED_TRACES_REMEMBERED:
        stopped_traces.popitem(last=False)


def current_trace() -> int:
    """The id of the trace this call belongs to, or 0 outside any trace."""
    monocle_trace = get_current_monocle_span().get_span_context().trace_id
    return monocle_trace or get_current_span().get_span_context().trace_id


def whole_call(args, kwargs) -> list:
    return [args, kwargs]


def response_as_returned(response):
    return response


def text_held_in(side: str, read: Callable, *read_from) -> str:
    """All the text and numbers in what ``read`` returns, or nothing if it cannot be read."""
    try:
        return "\n".join(texts_in(read(*read_from)))
    except Exception as e:
        logger.warning("Guardrails did not check the %s of a model call: %s", side, e)
        return ""


def texts_in(value, nesting: int = 0) -> Iterator[str]:
    """Every piece of text, and every number as text, held as data anywhere inside a value.

    Only data is looked into: mappings, lists, and objects that turn themselves into those. An SDK
    client, a stream or an HTTP response is not data and yields nothing.
    """
    if isinstance(value, str):
        yield value
    elif isinstance(value, bytes):
        yield value.decode("utf-8", errors="ignore")
    elif isinstance(value, bool):
        return
    elif isinstance(value, (int, float)):
        yield str(value)
    elif nesting < DEEPEST_NESTING_CHECKED:
        for part in parts_of(value):
            yield from texts_in(part, nesting + 1)


def parts_of(value) -> Iterable:
    if isinstance(value, Mapping):
        return value.values()
    if isinstance(value, (list, tuple, set)):
        return value
    if isinstance(value, type):
        return []
    if hasattr(value, "model_dump"):
        return [value.model_dump()]
    if dataclasses.is_dataclass(value):
        return [dataclasses.asdict(value)]
    return []
