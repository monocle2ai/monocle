"""Keep old import paths working after code moves, with a warning."""
import sys
import warnings
from importlib import import_module
from typing import Any, Dict


def alias_module(old_name: str, new_name: str) -> None:
    """Make the module at old_name be the module at new_name.

    Both names then point to the same module, so imports and mock.patch work through either.
    """
    _warn(old_name, new_name)
    sys.modules[old_name] = import_module(new_name)


def moved_attr(module_name: str, attr: str, moved: Dict[str, str]) -> Any:
    """Body of a module __getattr__: load a moved name from its new module.

    moved maps each old name to the module that now holds it.
    """
    if attr not in moved:
        raise AttributeError(f"module {module_name!r} has no attribute {attr!r}")
    _warn(f"{module_name}.{attr}", f"{moved[attr]}.{attr}")
    value = getattr(import_module(moved[attr]), attr)
    # Save it on the module so later lookups don't come back here.
    setattr(sys.modules[module_name], attr, value)
    return value


def _warn(old: str, new: str) -> None:
    warnings.warn(f"{old} has moved to {new}", DeprecationWarning, stacklevel=3)
