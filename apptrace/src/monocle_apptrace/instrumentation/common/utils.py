import ast
import logging, json
import os
import traceback
from typing import Callable, Generic, Optional, TypeVar, Mapping, Union

from opentelemetry.context import attach, detach, get_current, get_value, set_value, Context
from opentelemetry.trace import NonRecordingSpan, Span
from opentelemetry.trace.propagation import _SPAN_KEY
from opentelemetry.sdk.trace import id_generator, TracerProvider
from opentelemetry.propagate import extract
from opentelemetry import baggage
from monocle_apptrace.instrumentation.common.constants import MONOCLE_SCOPE_NAME_PREFIX, SCOPE_METHOD_FILE, SCOPE_CONFIG_PATH, llm_type_map, MONOCLE_SDK_VERSION, ADD_NEW_WORKFLOW
from importlib.metadata import version
from opentelemetry.trace.span import INVALID_SPAN
_MONOCLE_SPAN_KEY = "monocle" + _SPAN_KEY

T = TypeVar('T')
U = TypeVar('U')

logger = logging.getLogger(__name__)

embedding_model_context = {}
scope_id_generator = id_generator.RandomIdGenerator()
http_scopes:dict[str:str] = {}

try:
    monocle_sdk_version = version("monocle_apptrace")
except Exception as e:
    monocle_sdk_version = "unknown"
    logger.warning("Exception finding monocle-apptrace version.")

class MonocleSpanException(Exception):
    def __init__(self, err_message:str, err_code:str = None):
        """
        Monocle exeption to indicate error in span processing.        
        Parameters:
        - err_message (str): Error message.
        - status (str): Status code
        """
        super().__init__(err_message)
        self.message = err_message
        self.err_code = err_code

    def __str__(self):
        """String representation of the exception."""
        return f"[Monocle Span Error: {self.message}"

    def get_err_code(self):
        """Retrieve the error code."""
        return self.err_code

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
        def wrapper(wrapped, instance, args, kwargs, source_path=None):
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
            if not source_path:
                if traceback.extract_stack().__len__() > 2:
                    filename, line_number, _, _ = traceback.extract_stack()[-2]
                    source_path = f"{filename}:{line_number}"
                else:
                    source_path = ""
            val = func(tracer, handler, to_wrap, wrapped, instance, source_path, args, kwargs)
            return val

        return wrapper

    return _with_tracer


def resolve_from_alias(my_map, alias):
    """Find a alias that is not none from list of aliases"""

    for i in alias:
        if i in my_map.keys() and my_map[i] is not None:
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

def set_scope(scope_name: str, scope_value:str = None, context:Context = None) -> object:
    return set_scopes({scope_name: scope_value}, context)

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

def get_parent_span() -> Span:
    parent_span: Span = None
    _parent_span_context = get_current()
    if _parent_span_context is not None and _parent_span_context.get(_SPAN_KEY, None):
        parent_span = _parent_span_context.get(_SPAN_KEY, None)    
    return parent_span

def extract_http_headers(headers) -> object:
    global http_scopes
    trace_context:Context = extract(headers, context=get_current())
    trace_context = set_value(ADD_NEW_WORKFLOW, True, trace_context)
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
    try:
        if headers is not None:
            token = extract_http_headers(headers)
        return await func(*args, **kwargs)
    finally:
        if token is not None:
            clear_http_scopes(token)

# def run_async_with_scope(method, current_context, exceptions, *args, **kwargs):
#     token = None
#     try:
#         if current_context:
#             token = attach(current_context)
#         return asyncio.run(method(*args, **kwargs))
#     except Exception as e:
#         exceptions['exception'] = e
#         raise e
#     finally:
#         if token:
#             detach(token)

# async def async_wrapper(method, headers=None, *args, **kwargs):
#     current_context = get_current()
#     try:
#         if run_loop and run_loop.is_running():
#             results = []
#             thread = threading.Thread(target=lambda: results.append(run_async_with_scope(method, current_context, exceptions, *args, **kwargs)))
#             thread.start()
#             thread.join()
#             if 'exception' in exceptions:
#                 raise exceptions['exception']
#             return_value = results[0] if len(results) > 0 else None
#             return return_value
#         else:
#             return run_async_with_scope(method, None, exceptions, *args, **kwargs)
#     finally:
#         if token:
#             remove_scope(token)

def get_monocle_version() -> str:
    global monocle_sdk_version
    return monocle_sdk_version

def add_monocle_trace_state(headers:dict[str:str]) -> None:
    if headers is None:
        return
    monocle_trace_state = f"{MONOCLE_SDK_VERSION}={get_monocle_version()}"
    if 'tracestate' in headers:
        headers['tracestate'] = f"{headers['tracestate']},{monocle_trace_state}"
    else:
        headers['tracestate'] = monocle_trace_state

def get_json_dumps(obj) -> str:
    try:
        return json.dumps(obj)
    except TypeError as e:
        return str(obj)

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
            return Option(func(self.value))
        return Option(None)

# Example usage
def try_option(func: Callable[..., T], *args, **kwargs) -> Option[T]:
    try:
        return Option(func(*args, **kwargs))
    except Exception:
        return Option(None)

def get_llm_type(instance):
    try:
        t_name = type(instance).__name__.lower()
        t_name = t_name.replace("async", "") if "async" in t_name else t_name
        llm_type = llm_type_map.get(t_name)
        return llm_type
    except:
        pass

def get_status(arguments):
    if arguments['exception'] is not None:
        return 'error'
    else:
        return 'success'

def get_exception_status_code(arguments):
    if arguments['exception'] is not None and hasattr(arguments['exception'], 'code') and arguments['exception'].code is not None:
        return arguments['exception'].code
    elif arguments['exception'] is not None:
        return 'error'
    else:
        return 'success'

def get_exception_message(arguments):
    if arguments['exception'] is not None:
        if hasattr(arguments['exception'], 'message'):
            return arguments['exception'].message
        else:
            return arguments['exception'].__str__()
    else:
        return ''


def get_error_message(arguments):
    status_code = get_status_code(arguments)
    if status_code == 'success':
        return ''
    else:
        return status_code


def get_status_code(arguments):
    if "exception" in arguments and arguments["exception"] is not None:
        return get_exception_status_code(arguments)
    elif hasattr(arguments["result"], "status"):
        return arguments["result"].status
    else:
        return 'success'

def get_status(arguments):
    if arguments["exception"] is not None:
        return 'error'
    elif get_status_code(arguments) == 'success':
        return 'success'
    else:
        return 'error'

def patch_instance_method(obj, method_name, func):
    """
    Patch a special method (like __iter__) for a single instance.

    Args:
        obj: the instance to patch
        method_name: the name of the method (e.g., '__iter__')
        func: the new function, expecting (self, ...)
    """
    cls = obj.__class__
    # Dynamically create a new class that inherits from obj's class
    new_cls = type(f"Patched{cls.__name__}", (cls,), {
        method_name: func
    })
    obj.__class__ = new_cls


def set_monocle_span_in_context(
    span: Span, context: Optional[Context] = None
) -> Context:
    """Set the span in the given context.

    Args:
        span: The Span to set.
        context: a Context object. if one is not passed, the
            default current context is used instead.
    """
    ctx = set_value(_MONOCLE_SPAN_KEY, span, context=context)
    return ctx

def get_current_monocle_span(context: Optional[Context] = None) -> Span:
    """Retrieve the current span.

    Args:
        context: A Context object. If one is not passed, the
            default current context is used instead.

    Returns:
        The Span set in the context if it exists. INVALID_SPAN otherwise.
    """
    span = get_value(_MONOCLE_SPAN_KEY, context=context)
    if span is None or not isinstance(span, Span):
        return INVALID_SPAN
    return span

def get_input_event_from_span(events: list[dict], search_key:str) -> Optional[Mapping]:
    """Extract the 'data.input' event from the span if it exists.

    Args:
        span: The Span to extract the event from.

    Returns:
        The 'data.input' event if it exists, None otherwise.
    """
    input_request = None
    for event in events:
        if event.name == "data.input":
            try:
                #load the input attribute as dictionary from string
                try:
                    input_dict = json.loads(event.attributes.get("input", "{}"))
                except Exception as e:
                    if isinstance(e, json.JSONDecodeError):
                        input_dict = ast.literal_eval(event.attributes.get("input", {}))
                    else:
                        raise
                if search_key in input_dict:
                    input_request = input_dict[search_key]
            except Exception as e:
                logger.debug(f"Error parsing input event attribute: {e}")
            break
    return input_request

def replace_placeholders(obj: Union[dict, list, str], span: Span) -> Union[dict, list, str]:
    """Replace placeholders in strings with span context values."""
    if isinstance(obj, dict):
        return {k: replace_placeholders(v, span) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_placeholders(item, span) for item in obj]
    elif isinstance(obj, str):
        startIndex = 0
        while True:
            start = obj.find("{{", startIndex)
            end = obj.find("}}", start + 2)
            if start == -1 or end == -1:
                break
            key = obj[start + 2:end].strip()
            value = get_input_event_from_span(span.events, key)
            if value is not None:
                obj = obj[:start] + str(value) + obj[end + 2:]
            startIndex = start + len(str(value))
            if startIndex >= len(obj):
                break
        return obj
    else:
        return obj

