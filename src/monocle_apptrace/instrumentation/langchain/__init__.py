import os

from monocle_apptrace.instrumentation.common.utils import get_wrapper_methods_config

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LANGCHAIN_METHODS = get_wrapper_methods_config(
    wrapper_methods_config_path=os.path.join(os.path.dirname(__file__),'langchain_methods.json'),
    attributes_config_base_path=os.path.abspath(os.path.join(os.path.dirname(__file__), 'metamodel', 'attributes')))