import os
from monocle_apptrace.utils import load_wrapper_from_config

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LANGCHAIN_METHODS = load_wrapper_from_config(
    os.path.join(parent_dir, 'wrapper_config', 'lang_chain_methods.json'))
