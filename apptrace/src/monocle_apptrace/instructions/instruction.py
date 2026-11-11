import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from opentelemetry.context import get_value

from monocle_apptrace.instrumentation.common.constants import MONOCLE_INSTRUCTION, MONOCLE_INSTRUCTION_ENV
from monocle_apptrace.instrumentation.common.utils import fields_of_named_file

INSTRUCTION_FILE_KEYS = ("name", "description", "instruction")


@dataclass(frozen=True)
class Instruction:
    name: str
    text: str


def instruction_in_effect() -> Optional[Instruction]:
    """The instruction to add to model calls right now, or None if there is none.

    The path to its file is asked of the OpenTelemetry context first, then of the environment.
    """
    path = get_value(MONOCLE_INSTRUCTION) or os.environ.get(MONOCLE_INSTRUCTION_ENV)
    if not path:
        return None
    return load_instruction(path)


@lru_cache(maxsize=None)
def load_instruction(path: str) -> Instruction:
    """Read an instruction from its YAML file. Raises ValueError if the file is missing or not valid."""
    fields = fields_of_named_file(path, "Instruction", INSTRUCTION_FILE_KEYS)
    return Instruction(fields["name"], fields["instruction"].strip())
