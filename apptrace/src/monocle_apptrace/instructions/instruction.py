import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from opentelemetry.context import get_value

from monocle_apptrace.instrumentation.common.constants import MONOCLE_INSTRUCTION, MONOCLE_INSTRUCTION_ENV
from monocle_apptrace.providers import get_provider, provider_names

SHIPPED_INSTRUCTIONS = Path(__file__).parent
INSTRUCTION_FILE_KEYS = ("name", "description", "instruction")


@dataclass(frozen=True)
class Instruction:
    name: str
    text: str


def instruction_in_effect() -> Optional[Instruction]:
    """The instruction to add to model calls right now, or None if there is none.

    The OpenTelemetry context is asked first, then the environment.
    """
    name_or_path = get_value(MONOCLE_INSTRUCTION) or os.environ.get(MONOCLE_INSTRUCTION_ENV)
    if not name_or_path:
        return None
    return load_instruction(name_or_path)


@lru_cache(maxsize=None)
def load_instruction(name_or_path: str) -> Instruction:
    """Find an instruction by the name of a shipped one, or by the path to a YAML file.

    Raises ValueError if there is no such instruction or its file is not valid.
    """
    if is_yaml_path(name_or_path):
        return read_instruction_file(Path(name_or_path))
    if name_or_path in shipped_instruction_names():
        return read_instruction_file(SHIPPED_INSTRUCTIONS / f"{name_or_path}.yaml")
    text = text_from_providers(name_or_path)
    if text:
        return Instruction(name_or_path, text)
    raise ValueError(f"There is no instruction named '{name_or_path}'. "
                     f"Shipped instructions: {', '.join(shipped_instruction_names())}.")


def is_yaml_path(name_or_path: str) -> bool:
    return name_or_path.endswith((".yaml", ".yml"))


def text_from_providers(name: str) -> Optional[str]:
    for provider_name in provider_names():
        text = get_provider(provider_name).get_instruction(name)
        if text:
            return text
    return None


def read_instruction_file(path: Path) -> Instruction:
    if not path.is_file():
        raise ValueError(f"Instruction file {path} was not found.")
    import yaml
    fields = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(fields, dict):
        raise ValueError(f"Instruction file {path} must hold a name, a description and an instruction.")
    for key in INSTRUCTION_FILE_KEYS:
        if not isinstance(fields.get(key), str) or not fields[key].strip():
            raise ValueError(f"Instruction file {path} has no '{key}'.")
    if fields["name"] != path.stem:
        raise ValueError(f"Instruction file {path} is named '{fields['name']}'; the name must match the file name.")
    return Instruction(fields["name"], fields["instruction"].strip())


def shipped_instruction_names() -> List[str]:
    return sorted(path.stem for path in SHIPPED_INSTRUCTIONS.glob("*.yaml"))
