
import os
import logging
from monocle_apptrace.utils import load_wrapper_from_config

logger = logging.getLogger(__name__)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
HAYSTACK_METHODS = load_wrapper_from_config(
    os.path.join(parent_dir, 'metamodel', 'maps', 'haystack_methods.json'))
