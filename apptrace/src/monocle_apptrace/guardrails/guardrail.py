import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Pattern, Tuple

from monocle_apptrace.instrumentation.common.constants import (
    MONOCLE_GUARDRAILS_ENV, MONOCLE_GUARDRAILS_WATCH_ONLY_ENV,
)
from monocle_apptrace.instrumentation.common.utils import _normalize_bool, fields_of_named_file

INPUT = "input"
OUTPUT = "output"
SIDES = (INPUT, OUTPUT)
REQUIRED_TEXT_KEYS = ("name", "description")
GUARDRAIL_FILE_KEYS = ("name", "description", "sides", "patterns")


@dataclass(frozen=True)
class Trigger:
    """One guardrail triggering on one side of a model call."""
    guardrail: str
    side: str
    reason: str


@dataclass(frozen=True)
class Guardrail:
    name: str
    sides: Tuple[str, ...]
    patterns: Tuple[Pattern, ...]

    def trigger_on(self, side: str, text: str) -> Optional[Trigger]:
        """This guardrail triggering on the text of this side, or None if no pattern matches it.

        The reason names the pattern by its place in the file, never the text it matched.
        """
        for number, pattern in enumerate(self.patterns, start=1):
            if pattern.search(text):
                return Trigger(self.name, side, f"pattern {number}")
        return None


def guardrails_in_effect() -> List[Guardrail]:
    """The guardrails to check model calls against right now, in the order the environment lists their files."""
    paths = os.environ.get(MONOCLE_GUARDRAILS_ENV, "").split(",")
    return [load_guardrail(path.strip()) for path in paths if path.strip()]


def watching_only() -> bool:
    """True when a guardrail that triggers is recorded on the span but stops nothing."""
    return _normalize_bool(os.environ.get(MONOCLE_GUARDRAILS_WATCH_ONLY_ENV, ""))


@lru_cache(maxsize=None)
def load_guardrail(path: str) -> Guardrail:
    """Read a guardrail from its YAML file. Raises ValueError if the file is missing or not valid."""
    file = Path(path)
    fields = fields_of_named_file(path, "Guardrail", REQUIRED_TEXT_KEYS)
    for key in fields:
        if key not in GUARDRAIL_FILE_KEYS:
            raise ValueError(f"Guardrail file {file} has a key '{key}', which is not one of {', '.join(GUARDRAIL_FILE_KEYS)}.")
    return Guardrail(fields["name"], sides_checked(fields, file), compiled_patterns(fields, file))


def sides_checked(fields: dict, file: Path) -> Tuple[str, ...]:
    """The sides of the model call the file's 'sides' key lists. A file without one checks both."""
    sides = fields.get("sides", list(SIDES))
    if not isinstance(sides, list) or not sides:
        raise ValueError(f"Guardrail file {file} must list 'sides' as {INPUT}, {OUTPUT} or both.")
    for side in sides:
        if side not in SIDES:
            raise ValueError(f"Guardrail file {file} checks '{side}'; a guardrail checks {INPUT}, {OUTPUT} or both.")
    return tuple(sides)


def compiled_patterns(fields: dict, file: Path) -> Tuple[Pattern, ...]:
    patterns = fields.get("patterns")
    if not isinstance(patterns, list) or not patterns:
        raise ValueError(f"Guardrail file {file} has no 'patterns'.")
    return tuple(compiled_pattern(pattern, number, file) for number, pattern in enumerate(patterns, start=1))


def compiled_pattern(pattern, number: int, file: Path) -> Pattern:
    if not isinstance(pattern, str) or not pattern:
        raise ValueError(f"Guardrail file {file}: pattern {number} is not text.")
    try:
        compiled = re.compile(pattern)
    except re.error as e:
        raise ValueError(f"Guardrail file {file}: pattern {number} is not a valid regular expression: {e}.")
    if compiled.search(""):
        raise ValueError(f"Guardrail file {file}: pattern {number} matches empty text, so it would stop every model call.")
    return compiled
