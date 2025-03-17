import logging, json
import os
from typing import Callable, Generic, Optional, TypeVar, Mapping
import threading, asyncio

from opentelemetry.context import attach, detach, get_current, get_value, set_value, Context
from opentelemetry.trace import NonRecordingSpan, Span, get_tracer
from opentelemetry.trace.propagation import _SPAN_KEY
from opentelemetry.sdk.trace import id_generator, TracerProvider
from opentelemetry.propagate import inject, extract
from opentelemetry import baggage
from monocle_apptrace.instrumentation.common.constants import MONOCLE_SCOPE_NAME_PREFIX, SCOPE_METHOD_FILE, SCOPE_CONFIG_PATH, llm_type_map

T = TypeVar('T')
U = TypeVar('U')

logger = logging.getLogger(__name__)

monocle_tracer_provider: TracerProvider = None
embedding_model_context = {}
scope_id_generator = id_generator.RandomIdGenerator()
http_scopes:dict[str:str] = {}

class MonocleSpanException(Exception):
    def __init__(self, err_message:str):
        """
        Monocle exeption to indicate error in span processing.        
        Parameters:
        - err_message (str): Error message.
        - status (str): Status code
        """
        super().__init__(err_message)
        self.message = err_message

    def __str__(self):
        """String representation of the exception."""
        return f"[Monocle Span Error: {self.message} {self.status}"

def set_tracer_provider(tracer_provider: TracerProvider):
    global monocle_tracer_provider
    monocle_tracer_provider = tracer_provider

def get_tracer_provider() -> TracerProvider:
    global monocle_tracer_provider
    return monocle_tracer_provider

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
            try:
                # get and log the parent span context if injected by the application
                # This is useful for debugging and tracing of Azure functions
                _parent_span_context = get_current()
                if _parent_span_context is not None and _parent_span_context.get(_SPAN_KEY, None):
                    parent_span: Span = _parent_span_context.get(_SPAN_KEY, None)
                    is_span = isinstance(parent_span, NonRecordingSpan)
                    if is_span:
                        logger.debug(
                            f"Parent span is found with trace id {hex(parent_span.get_span_context().trace_id)}")
            except Exception as e:
                logger.error("Exception in attaching parent context: %s", e)

            val = func(tracer, handler, to_wrap, wrapped, instance, args, kwargs)
            return val

        return wrapper

    return _with_tracer

def resolve_from_alias(my_map, alias):
    """Find a alias that is not none from list of aliases"""

    for i in alias and my_map[i] is not None:
        if i in my_map.keys():
            return my_map[i]
    return None

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

def load_scopes() -> dict:
    methods_data = []
    scope_methods = []
    try:
        scope_config_file_path = os.environ.get(SCOPE_CONFIG_PATH, 
                                        os.path.join(os.getcwd(), SCOPE_METHOD_FILE))
        with open(scope_config_file_path) as f:
            methods_data = json.load(f)
            for method in methods_data:
                if method.get('http_header'):
                    http_scopes[method.get('http_header')] = method.get('scope_name')
                else:
                    scope_methods.append(method)
    except Exception as e:
        logger.debug(f"Error loading scope methods from file: {e}")
    return scope_methods

def __generate_scope_id() -> str:
    global scope_id_generator
    return f"{hex(scope_id_generator.generate_trace_id())}"

def set_scope(scope_name: str, scope_value:str = None) -> object:
    return set_scopes({scope_name: scope_value})

def set_scopes(scopes:dict[str, object], baggage_context:Context = None) -> object:
    if baggage_context is None:
        baggage_context:Context = get_current()
    for scope_name, scope_value in scopes.items():
        if scope_value is None:
            scope_value = __generate_scope_id()
        baggage_context = baggage.set_baggage(f"{MONOCLE_SCOPE_NAME_PREFIX}{scope_name}", scope_value, baggage_context)
        
    token:object = attach(baggage_context)
    return token

def remove_scope(token:object) -> None:
    remove_scopes(token)

def remove_scopes(token:object) -> None:
    if token is not None:
        detach(token)

def get_scopes() -> dict[str, object]:
    monocle_scopes:dict[str, object] = {}
    for key, val in baggage.get_all().items():
        if key.startswith(MONOCLE_SCOPE_NAME_PREFIX):
            monocle_scopes[key[len(MONOCLE_SCOPE_NAME_PREFIX):]] = val
    return monocle_scopes

def get_baggage_for_scopes():
    baggage_context:Context = None
    for scope_key, scope_value in get_scopes():
        monocle_scope_name = f"{MONOCLE_SCOPE_NAME_PREFIX}{scope_key}"
        baggage_context = baggage.set_baggage(monocle_scope_name, scope_value, context=baggage_context)
    return baggage_context

def set_scopes_from_baggage(baggage_context:Context):
    for scope_key, scope_value in baggage.get_all(baggage_context):
        if scope_key.startswith(MONOCLE_SCOPE_NAME_PREFIX):
            scope_name = scope_key[len(MONOCLE_SCOPE_NAME_PREFIX):]
            set_scope(scope_name, scope_value)

def extract_http_headers(headers) -> object:
    global http_scopes
    trace_context:Context = extract(headers, context=get_current())
    imported_scope:dict[str, object] = {}
    for http_header, http_scope in http_scopes.items():
        if http_header in headers:
            imported_scope[http_scope] = f"{http_header}: {headers[http_header]}"
    token = set_scopes(imported_scope, trace_context)
    return token

def clear_http_scopes(token:object) -> None:
    global http_scopes
    remove_scopes(token)

def http_route_handler(func, *args, **kwargs):
    if 'req' in kwargs and hasattr(kwargs['req'], 'headers'):
        headers = kwargs['req'].headers
    else:
        headers = None
    token = None
    if headers is not None:
        token = extract_http_headers(headers)
    try:
        result = func(*args, **kwargs)
        return result
    finally:
        if token is not None:
            clear_http_scopes(token)

async def http_async_route_handler(func, *args, **kwargs):
    if 'req' in kwargs and hasattr(kwargs['req'], 'headers'):
        headers = kwargs['req'].headers
    else:
        headers = None
    return async_wrapper(func, None, None, headers, *args, **kwargs)

def run_async_with_scope(method, current_context, exceptions, *args, **kwargs):
    token = None
    try:
        if current_context:
            token = attach(current_context)
        return asyncio.run(method(*args, **kwargs))
    except Exception as e:
        exceptions['exception'] = e
        raise e
    finally:
        if token:
            detach(token)

def async_wrapper(method, scope_name=None, scope_value=None, headers=None, *args, **kwargs):
    try:
        run_loop = asyncio.get_running_loop()
    except RuntimeError:
        run_loop = None

    token = None
    exceptions = {}
    if scope_name:
        token = set_scope(scope_name, scope_value)
    elif headers:
        token = extract_http_headers(headers)
    current_context = get_current()
    try:
        if run_loop and run_loop.is_running():
            results = []
            thread = threading.Thread(target=lambda: results.append(run_async_with_scope(method, current_context, exceptions, *args, **kwargs)))
            thread.start()
            thread.join()
            if 'exception' in exceptions:
                raise exceptions['exception']
            return_value = results[0] if len(results) > 0 else None
            return return_value
        else:
            return run_async_with_scope(method, None, exceptions, *args, **kwargs)
    finally:
        if token:
            remove_scope(token)

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

def get_llm_type(instance):
    try:
        llm_type = llm_type_map.get(type(instance).__name__.lower())
        return llm_type
    except:
        pass

def resolve_from_alias(my_map, alias):
    """Find a alias that is not none from list of aliases"""
    for i in alias:
        if i in my_map.keys():
            return my_map[i]
    return None