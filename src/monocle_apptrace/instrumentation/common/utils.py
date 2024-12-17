import logging
from typing import Callable, Generic, Optional, TypeVar

from opentelemetry.context import attach, detach, get_current, get_value, set_value
from opentelemetry.trace import NonRecordingSpan, Span
from opentelemetry.trace.propagation import _SPAN_KEY

T = TypeVar('T')
U = TypeVar('U')

logger = logging.getLogger(__name__)

embedding_model_context = {}

def set_span_attribute(span, name, value):
    if value is not None:
        if value != "":
            span.set_attribute(name, value)

def dont_throw(func):
    """
    A decorator that wraps the passed in function and logs exceptions instead of throwing them.

    @param func: The function to wrap
    @return: The wrapper function
    """
    # Obtain a logger specific to the function's module
    logger = logging.getLogger(func.__module__)

    # pylint: disable=inconsistent-return-statements
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as ex:
            logger.warning("Failed to execute %s, error: %s", func.__name__, str(ex))

    return wrapper

def with_tracer_wrapper(func):
    """Helper for providing tracer for wrapper functions."""

    def _with_tracer(tracer, handler, to_wrap):
        def wrapper(wrapped, instance, args, kwargs):
            token = None
            try:
                _parent_span_context = get_current()
                if _parent_span_context is not None and _parent_span_context.get(_SPAN_KEY, None):
                    parent_span: Span = _parent_span_context.get(_SPAN_KEY, None)
                    is_invalid_span = isinstance(parent_span, NonRecordingSpan)
                    if is_invalid_span:
                        token = attach(context={})
            except Exception as e:
                logger.error("Exception in attaching parent context: %s", e)

            val = func(tracer, handler, to_wrap, wrapped, instance, args, kwargs)
            # Detach the token if it was set
            if token:
                try:
                    detach(token=token)
                except Exception as e:
                    logger.error("Exception in detaching parent context: %s", e)
            return val
        return wrapper

    return _with_tracer

def set_embedding_model(model_name: str):
    """
    Sets the embedding model in the global context.

    @param model_name: The name of the embedding model to set
    """
    embedding_model_context['embedding_model'] = model_name

def get_embedding_model() -> str:
    """
    Retrieves the embedding model from the global context.

    @return: The name of the embedding model, or 'unknown' if not set
    """
    return embedding_model_context.get('embedding_model', 'unknown')

def set_attribute(key: str, value: str):
    """
    Set a value in the global context for a given key.

    Args:
        key: The key for the context value to set.
        value: The value to set for the given key.
    """
    attach(set_value(key, value))

def get_attribute(key: str) -> str:
    """
    Retrieve a value from the global context for a given key.

    Args:
        key: The key for the context value to retrieve.

    Returns:
        The value associated with the given key.
    """
    return get_value(key)

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def get_fully_qualified_class_name(instance):
    if instance is None:
        return None
    module_name = instance.__class__.__module__
    qualname = instance.__class__.__qualname__
    return f"{module_name}.{qualname}"

# returns json path like key probe in a dictionary
def get_nested_value(data, keys):
    for key in keys:
        if isinstance(data, dict) and key in data:
            data = data[key]
        elif hasattr(data, key):
            data = getattr(data, key)
        else:
            return None
    return data


def get_keys_as_tuple(dictionary, *keys):
    return tuple(next((value for key, value in dictionary.items() if key.endswith(k) and value is not None), None) for k in keys)


class Option(Generic[T]):
    def __init__(self, value: Optional[T]):
        self.value = value

    def is_some(self) -> bool:
        return self.value is not None

    def is_none(self) -> bool:
        return self.value is None

    def unwrap_or(self, default: T) -> T:
        return self.value if self.is_some() else default

    def map(self, func: Callable[[T], U]) -> 'Option[U]':
        if self.is_some():
            return Option(func(self.value))
        return Option(None)

    def and_then(self, func: Callable[[T], 'Option[U]']) -> 'Option[U]':
        if self.is_some():
            return func(self.value)
        return Option(None)

# Example usage
def try_option(func: Callable[..., T], *args, **kwargs) -> Option[T]:
    try:
        return Option(func(*args, **kwargs))
    except Exception:
        return Option(None)