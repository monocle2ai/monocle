import logging
import json
from importlib import import_module
import os
from opentelemetry.trace import Span
from opentelemetry.context import attach, set_value, get_value
from monocle_apptrace.constants import azure_service_map, aws_service_map
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
            return func(tracer, to_wrap, wrapped, instance, args, kwargs)

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

def update_span_with_infra_name(span: Span, span_key: str):
    for key, val in azure_service_map.items():
        if key in os.environ:
            span.set_attribute(span_key, val)
    for key, val in aws_service_map.items():
        if key in os.environ:
            span.set_attribute(span_key, val)

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

def get_workflow_name(span: Span) -> str:
    try:
        return get_value("workflow_name") or span.resource.attributes.get("service.name")
    except Exception as e:
        logger.exception(f"Error getting workflow name: {e}")
        return None