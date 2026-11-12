"""Instructions: text Monocle adds to every model call, on top of the developer's prompt."""
from monocle_apptrace.instructions.instruction import Instruction, instruction_in_effect, load_instruction

__all__ = ["Instruction", "instruction_in_effect", "load_instruction"]
