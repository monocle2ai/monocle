"""Sending a model call with the instruction added on top of what the developer wrote."""
import inspect
import logging
from typing import Callable, List, Optional

from monocle_apptrace.instructions.instruction import Instruction, instruction_in_effect
from monocle_apptrace.instructions.kill_switch import KillSwitchTriggered, Trigger
from monocle_apptrace.instructions.stop_when import reason_to_stop
from monocle_apptrace.instrumentation.common.constants import (
    INSTRUCTION_INPUT_LABEL, SENT_INSTRUCTION, SPAN_TYPE_KEY, SPAN_TYPES,
)
from monocle_apptrace.instrumentation.common.utils import get_json_dumps

logger = logging.getLogger(__name__)

instructions_warned_about_unstopped_calls = set()


class InstructedCall:
    """A model method that adds an instruction to the call before sending it.

    When the instruction has a stop condition and the model replies that it is met, the reply is not
    handed back: KillSwitchTriggered is raised in its place.
    """

    def __init__(self, model_method: Callable, inference: dict, instruction: Instruction):
        self.model_method = model_method
        self.inference = inference
        self.instruction = instruction
        self.sent_instruction: Optional[str] = None

    def __call__(self, *args, **kwargs):
        stoppable = self.can_be_stopped(kwargs)
        if self.instruction.stop_when and not stoppable:
            warn_once_that_some_calls_are_not_stopped(self.instruction)
        instructed_kwargs = self.with_text_added(kwargs, self.text_to_send(with_stop_rule=stoppable))
        if instructed_kwargs is None:
            return self.model_method(*args, **kwargs)
        self.sent_instruction = self.instruction.name
        reply = self.model_method(*args, **instructed_kwargs)
        if stoppable:
            return self.unless_stopped(reply)
        return reply

    def can_be_stopped(self, kwargs: dict) -> bool:
        """Whether the stop rule goes with this call.

        It does when the instruction has a stop condition and the reply can be read before the app
        receives it: the call's SDK says where its reply text is, and the call does not ask for a stream.
        """
        if not self.instruction.stop_when:
            return False
        if "reply_text" not in self.inference or "asks_for_a_stream" not in self.inference:
            return False
        return not self.inference["asks_for_a_stream"](kwargs)

    def text_sent_with(self, kwargs: dict) -> Optional[str]:
        """The text this call is sent with on top of the developer's prompt, or None if none can be added to it."""
        text = self.text_to_send(with_stop_rule=self.can_be_stopped(kwargs))
        if self.with_text_added(kwargs, text) is None:
            return None
        return text

    def text_to_send(self, with_stop_rule: bool) -> str:
        if with_stop_rule:
            return self.instruction.text_with_stop_rule()
        return self.instruction.text

    def with_text_added(self, kwargs: dict, text: str) -> Optional[dict]:
        """The call's keyword arguments with the text added, or None if there is none or it cannot be added to this call."""
        if not text:
            return None
        try:
            return self.inference["add_instruction"](kwargs, text)
        except Exception as e:
            logger.warning("Instruction '%s' was not added to the model call: %s", self.instruction.name, e)
            return None

    def unless_stopped(self, reply):
        """The reply, unless the model says in it that the stop condition is met. KillSwitchTriggered is raised then.

        A reply that must be awaited is read once it has been.
        """
        if inspect.isawaitable(reply):
            return self.unless_stopped_once_awaited(reply)
        reason = self.reason_to_stop_given_in(reply)
        if reason is None:
            return reply
        raise KillSwitchTriggered(Trigger(instruction=self.instruction.name, check="stop_when", reason=reason))

    async def unless_stopped_once_awaited(self, pending_reply):
        return self.unless_stopped(await pending_reply)

    def reason_to_stop_given_in(self, reply) -> Optional[str]:
        try:
            return reason_to_stop(self.inference["reply_text"](reply))
        except Exception as e:
            logger.warning("Instruction '%s' did not check a model reply for its stop condition: %s", self.instruction.name, e)
            return None


def warn_once_that_some_calls_are_not_stopped(instruction: Instruction) -> None:
    if instruction.name in instructions_warned_about_unstopped_calls:
        return
    instructions_warned_about_unstopped_calls.add(instruction.name)
    logger.warning(
        "Instruction '%s': 'stop_when' is not supported on streamed model calls, or on SDK calls whose reply "
        "Monocle does not read. Such calls are not stopped.", instruction.name)


def instructed_call(to_wrap: dict, model_method: Callable) -> Callable:
    """The method to call at this wrap site: one that adds the instruction in effect, or the method itself."""
    output_processor = to_wrap.get("output_processor")
    if not isinstance(output_processor, dict) or "add_instruction" not in output_processor:
        return model_method
    try:
        instruction = instruction_in_effect()
    except Exception as e:
        logger.warning("No instruction was added to the model call: %s", e)
        return model_method
    if instruction is None:
        return model_method
    return InstructedCall(model_method, output_processor, instruction)


def record_sent_instruction(model_method: Callable, span) -> None:
    if isinstance(model_method, InstructedCall) and model_method.sent_instruction:
        span.set_attribute(SENT_INSTRUCTION, model_method.sent_instruction)


def input_with_instruction(input_messages: List[str], span, model_method: Callable, kwargs: dict) -> List[str]:
    """A span's input messages, with the instruction its call carries listed first under its own label."""
    text = text_carried_by(span, model_method, kwargs)
    if text is None:
        return input_messages
    added_by_an_outer_layer = get_json_dumps({"system": text})
    developer_messages = [message for message in input_messages if message != added_by_an_outer_layer]
    return [get_json_dumps({INSTRUCTION_INPUT_LABEL: text}), *developer_messages]


def text_carried_by(span, model_method: Callable, kwargs: dict) -> Optional[str]:
    """The text this span's call was sent with on top of the developer's prompt, or None.

    A model SDK call carries it when it can be added to that call. A framework's inference span
    carries the one in effect, because the model SDK call beneath it is the one that adds it.
    """
    if isinstance(model_method, InstructedCall):
        return model_method.text_sent_with(kwargs)
    if span.attributes.get(SPAN_TYPE_KEY) != SPAN_TYPES.INFERENCE_FRAMEWORK:
        return None
    instruction = instruction_in_effect()
    if instruction is None:
        return None
    return instruction.text_with_stop_rule()


def add_system_message(kwargs: dict, text: str) -> Optional[dict]:
    """For a chat call that takes ``messages``: the call with the instruction as its first system message.

    Returns None if the call has no ``messages`` keyword to add to.
    """
    messages = kwargs.get("messages")
    if not messages:
        return None
    return {**kwargs, "messages": placed_first({"role": "system", "content": text}, list(messages))}


def placed_first(item, items: list) -> list:
    """The items with this one first, unless it already is first."""
    if items[:1] == [item]:
        return items
    return [item, *items]


def instruction_before(text: str, system_prompt) -> Optional[str]:
    """The instruction followed by the developer's system prompt.

    Returns None if there is a system prompt and it is not text, so it is never replaced.
    """
    if not system_prompt:
        return text
    if not isinstance(system_prompt, str):
        return None
    if system_prompt == text or system_prompt.startswith(f"{text}\n\n"):
        return system_prompt
    return f"{text}\n\n{system_prompt}"
