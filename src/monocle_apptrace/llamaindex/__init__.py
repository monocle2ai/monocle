
#pylint: disable=protected-access
import os
from monocle_apptrace.utils import load_wrapper_from_config

def get_llm_span_name_for_openai(instance):
    if (hasattr(instance, "_is_azure_client")
        and callable(getattr(instance, "_is_azure_client"))
        and instance._is_azure_client()):
        return "llamaindex.azure_openai"
    return "llamaindex.openai"

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LLAMAINDEX_METHODS = load_wrapper_from_config(
    os.path.join(parent_dir, 'wrapper_config', 'llama_index_methods.json'))
