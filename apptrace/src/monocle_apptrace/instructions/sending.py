"""Sending a model call with the instruction added on top of what the developer wrote."""
import logging
from typing import Callable, List, Optional

from monocle_apptrace.instructions.instruction import Instruction, instruction_in_effect
from monocle_apptrace.instrumentation.common.constants import (
    INSTRUCTION_INPUT_LABEL, SENT_INSTRUCTION, SPAN_TYPE_KEY, SPAN_TYPES,
)
from monocle_apptrace.instrumentation.common.utils import get_json_dumps

logger = logging.getLogger(__name__)


class InstructedCall:
    """A model method that adds an instruction to the call before sending it."""

    def __init__(self, model_method: Callable, add_instruction: Callable, instruction: Instruction):
        self.model_method = model_method
        self.add_instruction = add_instruction
        self.instruction = instruction
        self.sent_instruction: Optional[str] = None

    def __call__(self, *args, **kwargs):
        instructed_kwargs = self.instructed(kwargs)
        if instructed_kwargs is None:
            return self.model_method(*args, **kwargs)
        self.sent_instruction = self.instruction.name
        return self.model_method(*args, **instructed_kwargs)

    def instructed(self, kwargs: dict) -> Optional[dict]:
        """The call's keyword arguments with the instruction added, or None if it cannot be added to this call."""
        try:
            return self.add_instruction(kwargs, self.instruction.text)
        except Exception as e:
            logger.warning("Instruction '%s' was not added to the model call: %s", self.instruction.name, e)
            return None


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
    return InstructedCall(model_method, output_processor["add_instruction"], instruction)


def record_sent_instruction(model_method: Callable, span) -> None:
    if isinstance(model_method, InstructedCall) and model_method.sent_instruction:
        span.set_attribute(SENT_INSTRUCTION, model_method.sent_instruction)


def input_with_instruction(input_messages: List[str], span, model_method: Callable, kwargs: dict) -> List[str]:
    """A span's input messages, with the instruction its call carries listed first under its own label."""
    instruction = instruction_carried_by(span, model_method, kwargs)
    if instruction is None:
        return input_messages
    added_by_an_outer_layer = get_json_dumps({"system": instruction.text})
    developer_messages = [message for message in input_messages if message != added_by_an_outer_layer]
    return [get_json_dumps({INSTRUCTION_INPUT_LABEL: instruction.text}), *developer_messages]


def instruction_carried_by(span, model_method: Callable, kwargs: dict) -> Optional[Instruction]:
    """The instruction this span's call carries, or None.

    A model SDK call carries it when it can be added to that call. A framework's inference span
    carries the one in effect, because the model SDK call beneath it is the one that adds it.
    """
    if isinstance(model_method, InstructedCall):
        if model_method.instructed(kwargs) is None:
            return None
        return model_method.instruction
    if span.attributes.get(SPAN_TYPE_KEY) != SPAN_TYPES.INFERENCE_FRAMEWORK:
        return None
    return instruction_in_effect()
