import importlib.util
import logging

from monocle_apptrace.instrumentation.common.wrapper import task_wrapper, atask_wrapper
from monocle_apptrace.instrumentation.metamodel.mistral.entities.inference import (
    MISTRAL_INFERENCE,
    MISTRAL_STREAM_INFERENCE,
)
from monocle_apptrace.instrumentation.metamodel.mistral.entities.retrieval import MISTRAL_RETRIEVAL

logger = logging.getLogger(__name__)


def _resolve_package(candidates):
    """Return first importable module from candidates, handling mistralai SDK version differences."""
    for name in candidates:
        try:
            if importlib.util.find_spec(name) is not None:
                return name
        except (ImportError, AttributeError, ValueError):
            # e.g. a parent that is a module (not a package) -> submodule can't exist
            continue
    return candidates[0]


CHAT_PACKAGE = _resolve_package(["mistralai.chat", "mistralai.client.chat"])
EMBEDDINGS_PACKAGE = _resolve_package(["mistralai.embeddings", "mistralai.client.embeddings"])
logger.debug("Mistral instrumentation packages: chat=%s embeddings=%s", CHAT_PACKAGE, EMBEDDINGS_PACKAGE)

MISTRAL_METHODS = [
    {
        "package": CHAT_PACKAGE,               # where Chat is defined
        "object": "Chat",                      # class name
        "method": "complete",                  # the sync method
        "span_handler": "non_framework_handler",
        "wrapper_method": task_wrapper,
        "output_processor": MISTRAL_INFERENCE
    },
    {
        "package": CHAT_PACKAGE,               # where Chat is defined
        "object": "Chat",                      # class name
        "method": "complete_async",            # the async method
        "span_handler": "non_framework_handler",
        "wrapper_method": atask_wrapper,
        "output_processor": MISTRAL_INFERENCE
    },
    {
        "package": CHAT_PACKAGE,
        "object": "Chat",
        "method": "stream",              # sync streaming
        "span_handler": "non_framework_handler",
        "wrapper_method": task_wrapper,
        "output_processor": MISTRAL_STREAM_INFERENCE,
    },
    {
        "package": CHAT_PACKAGE,
        "object": "Chat",
        "method": "stream_async",        # async streaming
        "span_handler": "non_framework_handler",
        "wrapper_method": atask_wrapper,
        "output_processor": MISTRAL_STREAM_INFERENCE,
    },
    {
        "package": EMBEDDINGS_PACKAGE,         # where Embeddings is defined
        "object": "Embeddings",                # sync embeddings client
        "method": "create",                    # sync create
        "span_handler": "non_framework_handler",
        "wrapper_method": task_wrapper,
        "output_processor": MISTRAL_RETRIEVAL
    },
]

 


