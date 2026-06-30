from monocle_apptrace.instrumentation.common.wrapper import atask_wrapper, task_wrapper
from monocle_apptrace.instrumentation.metamodel.litellm.entities.inference import INFERENCE

LITELLM_METHODS = [
    {
        "package": "litellm.llms.openai.openai",
        "object": "OpenAIChatCompletion",
        "method": "completion",
        "wrapper_method": task_wrapper,
        "output_processor": INFERENCE,
        "span_handler": "litellm_sync_handler"
    },
    {
        "package": "litellm.llms.azure.azure",
        "object": "AzureChatCompletion",
        "method": "completion",
        "wrapper_method": task_wrapper,
        "output_processor": INFERENCE,
        "span_handler": "litellm_sync_handler"
    },
    {
        "package": "litellm.llms.openai.openai",
        "object": "OpenAIChatCompletion",
        "method": "acompletion",
        "wrapper_method": atask_wrapper,
        "output_processor": INFERENCE
    },
    {
        "package": "litellm.llms.azure.azure",
        "object": "AzureChatCompletion",
        "method": "acompletion",
        "wrapper_method": atask_wrapper,
        "output_processor": INFERENCE
    },
    {
        # Gemini (Google AI Studio) and Vertex AI both route through VertexLLM.
        "package": "litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini",
        "object": "VertexLLM",
        "method": "completion",
        "wrapper_method": task_wrapper,
        "output_processor": INFERENCE,
        "span_handler": "litellm_sync_handler"
    },
    {
        "package": "litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini",
        "object": "VertexLLM",
        "method": "async_completion",
        "wrapper_method": atask_wrapper,
        "output_processor": INFERENCE
    },
    {
        # AWS Bedrock — Claude, Nova, Llama, etc. route through the Converse API.
        "package": "litellm.llms.bedrock.chat.converse_handler",
        "object": "BedrockConverseLLM",
        "method": "completion",
        "wrapper_method": task_wrapper,
        "output_processor": INFERENCE,
        "span_handler": "litellm_sync_handler"
    },
    {
        "package": "litellm.llms.bedrock.chat.converse_handler",
        "object": "BedrockConverseLLM",
        "method": "async_completion",
        "wrapper_method": atask_wrapper,
        "output_processor": INFERENCE
    },
    {
        # Anthropic — Claude models via the Anthropic API. The sync `completion` is
        # the entry point; it dispatches to `acompletion_function` when acompletion=True.
        "package": "litellm.llms.anthropic.chat.handler",
        "object": "AnthropicChatCompletion",
        "method": "completion",
        "wrapper_method": task_wrapper,
        "output_processor": INFERENCE,
        "span_handler": "litellm_sync_handler"
    },
    {
        "package": "litellm.llms.anthropic.chat.handler",
        "object": "AnthropicChatCompletion",
        "method": "acompletion_function",
        "wrapper_method": atask_wrapper,
        "output_processor": INFERENCE
    }
]
