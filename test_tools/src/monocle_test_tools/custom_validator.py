"""Custom validators: the test author's own pass/fail check on a recorded response.

An eval grades a response against an expected label. A validator answers a
simpler question -- "is this output valid?" -- with the user's own logic. This
module holds the placeholder the framework calls; see ``check_validator``.
"""
import importlib
import inspect
from typing import Any, Callable, Optional, Union

from pydantic import BaseModel, ConfigDict

# What a validator returns: True (valid), False (invalid), or a message saying why.
ValidatorVerdict = Union[bool, str]


class BaseValidator(BaseModel):
    """Base for validators that carry options or state. A plain function needs neither."""
    validator_options: Optional[dict] = {}

    def validate_response(self, input: Optional[str] = None,  # pylint: disable=redefined-builtin
                          output: Optional[str] = None) -> ValidatorVerdict:
        """Judge one response.

        Args:
            input: What was asked, as recorded on the span.
            output: What was answered. None when the span recorded no response.

        Returns:
            True if valid, False if not, or a message explaining why not.
        """
        raise NotImplementedError

    @property
    def name(self) -> str:
        """What this validator is called in a failure message."""
        return type(self).__name__


class _CallableValidator(BaseValidator):
    """Adapts a plain function to the BaseValidator interface."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    func: Callable
    ref: Optional[str] = None

    def validate_response(self, input: Optional[str] = None,  # pylint: disable=redefined-builtin
                          output: Optional[str] = None) -> ValidatorVerdict:
        return self.func(input=input, output=output)

    @property
    def name(self) -> str:
        return self.ref or getattr(self.func, "__name__", None) or repr(self.func)


# How a validator may be named. Declared after BaseValidator so the alias holds the
# class itself: FluentTestCase.validators is annotated with it, and a forward
# reference would have to resolve in testcase.py's namespace instead of this one.
ValidatorRef = Union[str, Callable, BaseValidator]


def _check_signature(func: Callable, ref: Optional[str] = None) -> None:
    """Reject a function that cannot be called as ``func(input=..., output=...)``.

    Checked when the test names the validator, so a mistyped parameter fails with
    a message about the signature rather than a bare TypeError mid-run.
    """
    if inspect.iscoroutinefunction(func):
        raise ValueError(
            f"validator '{ref or getattr(func, '__name__', func)}' is async; validators are "
            "called synchronously, so an async one would only ever return a coroutine. Make "
            "it a plain function -- it is judging a response that was already recorded.")
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return  # A builtin exposes no signature; let the call itself be the judge.
    try:
        signature.bind(input=None, output=None)
    except TypeError as exc:
        raise ValueError(
            f"validator '{ref or getattr(func, '__name__', func)}' must be callable as "
            f"func(input=..., output=...), but its signature is {signature}: {exc}"
        ) from exc


def _import_validator(ref: str) -> Any:
    """Import the validator named by ``"package.module:attribute"`` (or a last dot)."""
    module_name, separator, attribute = ref.partition(":")
    if not separator:
        module_name, _, attribute = ref.rpartition(".")
    if not module_name or not attribute:
        raise ValueError(
            f"'{ref}' does not name an importable validator; write it as "
            "'package.module:function'")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ValueError(
            f"cannot import module '{module_name}' for validator '{ref}': {exc}") from exc
    except Exception as exc:
        # The module was found but blew up while executing. Name the reference, or the
        # traceback looks like it came from nowhere.
        raise ValueError(
            f"importing module '{module_name}' for validator '{ref}' failed: "
            f"{type(exc).__name__}: {exc}") from exc
    try:
        return getattr(module, attribute)
    except AttributeError as exc:
        raise ValueError(
            f"module '{module_name}' has no '{attribute}' for validator '{ref}'") from exc


def get_validator(validator: ValidatorRef) -> BaseValidator:
    """Resolve any way of naming a validator into one that can be called.

    Args:
        validator: A function, a BaseValidator instance or subclass, or an import
            path to any of those (``"my_pkg.checks:valid_order"``).

    Returns:
        A BaseValidator ready to judge responses.

    Raises:
        ValueError: If the import path cannot be resolved, or the function cannot
            be called as ``func(input=..., output=...)``.
        TypeError: If *validator* is not callable at all.
    """
    ref = validator if isinstance(validator, str) else None
    if ref is not None:
        validator = _import_validator(ref)

    if isinstance(validator, type) and issubclass(validator, BaseValidator):
        validator = validator()
    if isinstance(validator, BaseValidator):
        if inspect.iscoroutinefunction(validator.validate_response):
            raise ValueError(
                f"validator '{validator.name}' has an async validate_response; validators are "
                "called synchronously. Make it a plain method.")
        return validator
    if callable(validator):
        _check_signature(validator, ref)
        return _CallableValidator(func=validator, ref=ref)
    raise TypeError(
        f"a validator must be a function, a BaseValidator, or an import path to "
        f"either; got {type(validator).__name__}")


def read_verdict(verdict: ValidatorVerdict, *, name: str) -> Optional[str]:
    """Turn a validator's return value into a failure message, or None if it passed.

    Raises:
        TypeError: If the validator returned anything else. That is a bug in the
            validator, so it must not count as a pass or a fail.
    """
    if verdict is True:
        return None
    if verdict is False:
        return f"validator '{name}' rejected the response"
    if isinstance(verdict, str):
        return verdict.strip() or f"validator '{name}' rejected the response"
    if verdict is None:
        raise TypeError(
            f"validator '{name}' returned None; return True when the response is "
            "valid, and False or a message explaining the problem when it is not")
    raise TypeError(
        f"validator '{name}' returned {type(verdict).__name__}; a validator must "
        "return True, False, or a message explaining the problem")
