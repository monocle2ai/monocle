from monocle_apptrace.instrumentation.common.wrapper import task_wrapper
from monocle_apptrace.instrumentation.metamodel.gemini.entities.inference import (
    INFERENCE,
)
from monocle_apptrace.instrumentation.metamodel.gemini.entities.retrieval import (
    RETRIEVAL,
)

GEMINI_METHODS = [
    {
      "package": "google.genai.models",
      "object": "Models",
      "method": "generate_content",
      "wrapper_method": task_wrapper,
      "output_processor": INFERENCE,
    },
    {
      "package": "google.genai.models",
      "object": "Models",
      "method": "embed_content",
      "wrapper_method": task_wrapper,
      "output_processor": RETRIEVAL,
    }
]