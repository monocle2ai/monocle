import logging
import json
from importlib import import_module

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

def load_wrapper_from_config(config_file_path: str, module_name: str = None):
    wrapper_methods = []
    with open(config_file_path, encoding='UTF-8') as config_file:
        json_data = json.load(config_file)
        wrapper_methods = json_data["wrapper_methods"]
        for wrapper_method in wrapper_methods:
            wrapper_method["wrapper"] = get_wrapper_method(
                wrapper_method["wrapper_package"], wrapper_method["wrapper_method"])
            if "span_name_getter_method" in wrapper_method :
                wrapper_method["span_name_getter"] = get_wrapper_method(
                    wrapper_method["span_name_getter_package"],
                    wrapper_method["span_name_getter_method"])
        return wrapper_methods

def get_wrapper_method(package_name: str, method_name: str):
    wrapper_module = import_module("monocle_apptrace." + package_name)
    return getattr(wrapper_module, method_name)

def is_openai_endpoint(url: str):
    return "openai.com" in url

def is_azure_endpoint(url: str):
    return "azure.com" in url