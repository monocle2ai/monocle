import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

from opentelemetry.context import get_value

from monocle_apptrace.instructions.stop_when import stop_rule
from monocle_apptrace.instrumentation.common.constants import MONOCLE_INSTRUCTION, MONOCLE_INSTRUCTION_ENV

KEYS_EVERY_FILE_HOLDS = ("name", "description")
KEYS_SENT_TO_THE_MODEL = ("instruction", "stop_when")


@dataclass(frozen=True)
class Instruction:
    name: str
    text: str
    stop_when: str = ""

    def text_with_stop_rule(self) -> str:
        """The instruction text, then the stop rule. Either one is left out when the file does not hold it."""
        if not self.stop_when:
            return self.text
        if not self.text:
            return stop_rule(self.stop_when)
        return f"{self.text}\n\n{stop_rule(self.stop_when)}"


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
    """Read an instruction from its YAML file. Raises ValueError, naming the file, if it is missing or not valid."""
    file = Path(path)
    fields = fields_read_from(file)
    require_name_and_description(fields, file)
    require_instruction_or_stop_condition(fields, file)
    require_name_to_match_file_name(fields, file)
    return Instruction(fields["name"], fields.get("instruction", "").strip(), fields.get("stop_when", "").strip())


def fields_read_from(file: Path) -> dict:
    if not file.is_file():
        raise ValueError(f"Instruction file {file} was not found.")
    import yaml
    try:
        fields = yaml.safe_load(file.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ValueError(f"Instruction file {file} is not valid YAML: {e}")
    if not isinstance(fields, dict):
        raise ValueError(f"Instruction file {file} must hold a name, a description, and an instruction or a stop_when.")
    return fields


def require_name_and_description(fields: dict, file: Path) -> None:
    for key in KEYS_EVERY_FILE_HOLDS:
        if not is_text(fields.get(key)):
            raise ValueError(f"Instruction file {file} has no '{key}'.")


def require_instruction_or_stop_condition(fields: dict, file: Path) -> None:
    if not any(key in fields for key in KEYS_SENT_TO_THE_MODEL):
        raise ValueError(f"Instruction file {file} has no 'instruction' and no 'stop_when'. It must hold one or both.")
    for key in KEYS_SENT_TO_THE_MODEL:
        if key in fields and not is_text(fields[key]):
            raise ValueError(f"Instruction file {file}: '{key}' is empty or not text.")


def require_name_to_match_file_name(fields: dict, file: Path) -> None:
    if fields["name"] != file.stem:
        raise ValueError(f"Instruction file {file} is named '{fields['name']}'; the name must match the file name.")


def is_text(value) -> bool:
    return isinstance(value, str) and bool(value.strip())
