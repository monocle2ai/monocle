import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

from opentelemetry.context import get_value

from monocle_apptrace.instrumentation.common.constants import MONOCLE_INSTRUCTION, MONOCLE_INSTRUCTION_ENV

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
    file = Path(path)
    if not file.is_file():
        raise ValueError(f"Instruction file {file} was not found.")
    import yaml
    fields = yaml.safe_load(file.read_text(encoding="utf-8"))
    if not isinstance(fields, dict):
        raise ValueError(f"Instruction file {file} must hold a name, a description and an instruction.")
    for key in INSTRUCTION_FILE_KEYS:
        if not isinstance(fields.get(key), str) or not fields[key].strip():
            raise ValueError(f"Instruction file {file} has no '{key}'.")
    if fields["name"] != file.stem:
        raise ValueError(f"Instruction file {file} is named '{fields['name']}'; the name must match the file name.")
    return Instruction(fields["name"], fields["instruction"].strip())
