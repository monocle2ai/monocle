from monocle_apptrace.instrumentation.common.wrapper import atask_wrapper, task_wrapper
from monocle_apptrace.instrumentation.metamodel.openai.entities.inference import (
    INFERENCE,
)
from monocle_apptrace.instrumentation.metamodel.openai.entities.retrieval import (
    RETRIEVAL,
)

OPENAI_METHODS = [
    {
        "package": "openai.resources.chat.completions.completions",
        "object": "Completions",
        "method": "create",
        "wrapper_method": task_wrapper,
        "span_handler": "openai_handler",
        "output_processor": INFERENCE
    },
    {
        "package": "openai.resources.chat.completions.completions",
        "object": "AsyncCompletions",
        "method": "create",
        "wrapper_method": atask_wrapper,
        "span_handler": "openai_handler",
        "output_processor": INFERENCE
    },
    {
        "package": "openai.resources.embeddings",
        "object": "Embeddings",
        "method": "create",
        "wrapper_method": task_wrapper,
        "span_handler": "openai_handler",
        "output_processor": RETRIEVAL
    },
    {
        "package": "openai.resources.embeddings",
        "object": "AsyncEmbeddings",
        "method": "create",
        "wrapper_method": atask_wrapper,
        "span_handler": "openai_handler",
        "output_processor": RETRIEVAL
    },
    {
        "package": "openai.resources.responses.responses",
        "object": "Responses",
        "method": "create",
        "wrapper_method": task_wrapper,
        "span_handler": "openai_handler",
        "output_processor": INFERENCE
    },
    {
        "package": "openai.resources.responses.responses",
        "object": "AsyncResponses",
        "method": "create",
        "wrapper_method": atask_wrapper,
        "span_handler": "openai_handler",
        "output_processor": INFERENCE
    },
    # {
    #     "package": "openai.resources.responses.responses",
    #     "object": "ResponsesWithStreamingResponse",
    #     "method": "create",
    #     "wrapper_method": task_wrapper,
    #     "span_handler": "openai_handler",
    #     "output_processor": INFERENCE
    # },
    # {
    #     "package": "openai.resources.responses.responses",
    #     "object": "AsyncResponsesWithStreamingResponse",
    #     "method": "create",
    #     "wrapper_method": atask_wrapper,
    #     "span_handler": "openai_handler",
    #     "output_processor": INFERENCE
    # },
    # {
    #     "package": "openai.resources.responses.responses",
    #     "object": "ResponsesWithRawResponse",
    #     "method": "create",
    #     "wrapper_method": task_wrapper,
    #     "span_handler": "openai_handler",
    #     "output_processor": INFERENCE
    # },
    # {
    #     "package": "openai.resources.responses.responses",
    #     "object": "AsyncResponsesWithRawResponse",
    #     "method": "create",
    #     "wrapper_method": atask_wrapper,
    #     "span_handler": "openai_handler",
    #     "output_processor": INFERENCE
    # },
    {
        "package": "agents.models.openai_responses",
        "object": "OpenAIResponsesModel",
        "method": "_fetch_response",
        "wrapper_method": atask_wrapper,
        "span_handler": "openai_handler",
        "output_processor": INFERENCE
    },
    {
        "package": "agents.models.openai_responses",
        "object": "OpenAIResponsesWSModel",
        "method": "_fetch_response",
        "wrapper_method": atask_wrapper,
        "span_handler": "openai_handler",
        "output_processor": INFERENCE
    }

]