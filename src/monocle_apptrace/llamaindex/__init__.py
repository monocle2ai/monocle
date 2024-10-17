# pylint: disable=protected-access
import os
from monocle_apptrace.utils import get_wrapper_methods_config


def get_llm_span_name_for_openai(instance):
    if (hasattr(instance, "_is_azure_client")
            and callable(getattr(instance, "_is_azure_client"))
            and instance._is_azure_client()):
        return "llamaindex.azure_openai"
    return "llamaindex.openai"

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LLAMAINDEX_METHODS = get_wrapper_methods_config(
    wrapper_methods_config_path=os.path.join(parent_dir, 'metamodel', 'maps', 'llamaindex_methods.json'),
    attributes_config_base_path=os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
