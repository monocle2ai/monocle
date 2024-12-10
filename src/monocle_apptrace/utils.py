import logging
import json
from importlib import import_module
import os
from opentelemetry.trace import NonRecordingSpan,Span
from opentelemetry.trace.propagation import _SPAN_KEY
from opentelemetry.context import (attach, detach,get_current)
from opentelemetry.context import attach, set_value, get_value
from monocle_apptrace.constants import service_name_map, service_type_map
from json.decoder import JSONDecodeError

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

    def _with_tracer(tracer, to_wrap):
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

            val = func(tracer, to_wrap, wrapped, instance, args, kwargs)
            # Detach the token if it was set
            if token:
                try:
                    detach(token=token)
                except Exception as e:
                    logger.error("Exception in detaching parent context: %s", e)
            return val
        return wrapper

    return _with_tracer

def resolve_from_alias(my_map, alias):
    """Find a alias that is not none from list of aliases"""

    for i in alias:
        if i in my_map.keys():
            return my_map[i]
    return None

def load_output_processor(wrapper_method, attributes_config_base_path):
    """Load the output processor from a file if the file path is provided and valid."""
    logger = logging.getLogger()
    output_processor_file_path = wrapper_method["output_processor"][0]
    logger.info(f'Output processor file path is: {output_processor_file_path}')

    if isinstance(output_processor_file_path, str) and output_processor_file_path:  # Combined condition
        if not attributes_config_base_path:
            absolute_file_path = os.path.abspath(output_processor_file_path)
        else:
            absolute_file_path = os.path.join(attributes_config_base_path, output_processor_file_path)

        logger.info(f'Absolute file path is: {absolute_file_path}')
        try:
            with open(absolute_file_path, encoding='UTF-8') as op_file:
                wrapper_method["output_processor"] = json.load(op_file)
                logger.info('Output processor loaded successfully.')
        except FileNotFoundError:
            logger.error(f"Error: File not found at {absolute_file_path}.")
        except JSONDecodeError:
            logger.error(f"Error: Invalid JSON content in the file {absolute_file_path}.")
        except Exception as e:
            logger.error(f"Error: An unexpected error occurred: {e}")
    else:
        logger.error("Invalid or missing output processor file path.")

def get_wrapper_methods_config(
        wrapper_methods_config_path: str,
        attributes_config_base_path: str = None
):
    parent_dir = os.path.dirname(os.path.join(os.path.dirname(__file__), '..'))
    wrapper_methods_config = load_wrapper_methods_config_from_file(
        wrapper_methods_config_path=os.path.join(parent_dir, wrapper_methods_config_path))
    process_wrapper_method_config(
        wrapper_methods_config=wrapper_methods_config,
        attributes_config_base_path=attributes_config_base_path)
    return wrapper_methods_config

def load_wrapper_methods_config_from_file(
        wrapper_methods_config_path: str):
    json_data = {}

    with open(wrapper_methods_config_path, encoding='UTF-8') as config_file:
        json_data = json.load(config_file)

    return json_data["wrapper_methods"]

def process_wrapper_method_config(
        wrapper_methods_config: str,
        attributes_config_base_path: str = ""):
    for wrapper_method in wrapper_methods_config:
        if "wrapper_package" in wrapper_method and "wrapper_method" in wrapper_method:
            wrapper_method["wrapper"] = get_wrapper_method(
                wrapper_method["wrapper_package"], wrapper_method["wrapper_method"])
            if "span_name_getter_method" in wrapper_method:
                wrapper_method["span_name_getter"] = get_wrapper_method(
                    wrapper_method["span_name_getter_package"],
                    wrapper_method["span_name_getter_method"])
        if "output_processor" in wrapper_method and wrapper_method["output_processor"]:
            load_output_processor(wrapper_method, attributes_config_base_path)

def get_wrapper_method(package_name: str, method_name: str):
    wrapper_module = import_module("monocle_apptrace." + package_name)
    return getattr(wrapper_module, method_name)

def set_app_hosting_identifier_attribute(span, span_index):
    return_value = 0
    # Search env to indentify the infra service type, if found check env for service name if possible
    for type_env, type_name in service_type_map.items():
        if type_env in os.environ:
            return_value = 1
            span.set_attribute(f"entity.{span_index}.type", f"app_hosting.{type_name}")
            entity_name_env = service_name_map.get(type_name, "unknown")
            span.set_attribute(f"entity.{span_index}.name", os.environ.get(entity_name_env, "generic"))
    return return_value

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

def get_workflow_name(span: Span) -> str:
    try:
        return get_value("workflow_name") or span.resource.attributes.get("service.name")
    except Exception as e:
        logger.exception(f"Error getting workflow name: {e}")
        return None

def get_vectorstore_deployment(my_map):
    if isinstance(my_map, dict):
        if '_client_settings' in my_map:
            client = my_map['_client_settings'].__dict__
            host, port = get_keys_as_tuple(client, 'host', 'port')
            if host:
                return f"{host}:{port}" if port else host
        keys_to_check = ['client', '_client']
        host = get_host_from_map(my_map, keys_to_check)
        if host:
            return host
    else:
        if hasattr(my_map, 'client') and '_endpoint' in my_map.client.__dict__:
            return my_map.client.__dict__['_endpoint']
        host, port = get_keys_as_tuple(my_map.__dict__, 'host', 'port')
        if host:
            return f"{host}:{port}" if port else host
    return None

def get_keys_as_tuple(dictionary, *keys):
    return tuple(next((value for key, value in dictionary.items() if key.endswith(k) and value is not None), None) for k in keys)

def get_host_from_map(my_map, keys_to_check):
    for key in keys_to_check:
        seed_connections = get_nested_value(my_map, [key, 'transport', 'seed_connections'])
        if seed_connections and 'host' in seed_connections[0].__dict__:
            return seed_connections[0].__dict__['host']
    return None