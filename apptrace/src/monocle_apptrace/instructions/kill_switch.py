"""The kill switch: stopping a model call, and the agent that made it, with an error the trace can show."""
from dataclasses import dataclass
from typing import Optional

from monocle_apptrace.instrumentation.common.constants import (
    KILL_SWITCH_CHECK, KILL_SWITCH_ERROR_CODE, KILL_SWITCH_INSTRUCTION, KILL_SWITCH_REASON, SPAN_TYPE_KEY,
    SPAN_TYPES,
)


@dataclass(frozen=True)
class Trigger:
    """One check of an instruction file deciding that a model call must stop."""
    instruction: str
    check: str
    reason: str


class KillSwitchTriggered(Exception):
    """Raised in place of a model call, or of its reply, that the kill switch stopped."""
    code = KILL_SWITCH_ERROR_CODE

    def __init__(self, trigger: Trigger):
        super().__init__(f"Kill switch triggered by instruction '{trigger.instruction}': {trigger.reason}")
        self.trigger = trigger


def record_triggered_kill_switch(span, error: Optional[BaseException]) -> None:
    """Mark an inference span whose call failed because the kill switch stopped it, or stopped a model call beneath it."""
    trigger = trigger_that_caused(error)
    if trigger is None or not is_inference(span):
        return
    span.set_attribute(KILL_SWITCH_INSTRUCTION, trigger.instruction)
    span.set_attribute(KILL_SWITCH_CHECK, trigger.check)
    span.set_attribute(KILL_SWITCH_REASON, trigger.reason)


def is_inference(span) -> bool:
    if not span.is_recording():
        return False
    return str(span.attributes.get(SPAN_TYPE_KEY, "")).startswith(SPAN_TYPES.INFERENCE)


def trigger_that_caused(error: Optional[BaseException]) -> Optional[Trigger]:
    """The trigger behind this error, also when a library has wrapped a stopped call's error in its own."""
    already_looked_at = set()
    while error is not None and id(error) not in already_looked_at:
        if isinstance(error, KillSwitchTriggered):
            return error.trigger
        already_looked_at.add(id(error))
        error = error.__cause__ or error.__context__
    return None

