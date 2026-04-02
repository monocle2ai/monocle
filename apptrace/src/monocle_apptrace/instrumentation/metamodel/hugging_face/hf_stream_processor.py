"""HuggingFace Inference streaming processor.

HF AsyncInferenceClient.chat_completion(stream=True) yields
ChatCompletionStreamOutput chunks which follow the OpenAI-compatible
format (choices[0].delta.content, finish_reason, usage on final chunk).
"""

from monocle_apptrace.instrumentation.metamodel.openai.openai_stream_processor import (
    OpenAIStreamProcessor,
)


class HFStreamProcessor(OpenAIStreamProcessor):
    """Stream processor for HuggingFace InferenceClient chat_completion streaming.

    HF streaming uses the same OpenAI-compatible chunk format:
    - choices[0].delta.content for text tokens
    - choices[0].finish_reason on the final chunk
    - usage on the final chunk (when available)
    """
    pass
