"""HuggingFace Inference streaming processor.

HF AsyncInferenceClient.chat_completion(stream=True) yields
ChatCompletionStreamOutput chunks which follow the OpenAI-compatible
format (choices[0].delta.content, finish_reason, usage on final chunk).

Key difference from the OpenAI SDK: HF chunks are dataclass objects and
do NOT have an `.object` attribute, so the base OpenAIStreamProcessor
`handle_completion` check (`item.object == "chat.completion.chunk"`) never
matches.  We override it to use the presence of `.usage` instead.
"""

from typing import Any

from monocle_apptrace.instrumentation.common.stream_processor import StreamState
from monocle_apptrace.instrumentation.metamodel.openai.openai_stream_processor import (
    OpenAIStreamProcessor,
)


class HFStreamProcessor(OpenAIStreamProcessor):
    """Stream processor for HuggingFace InferenceClient chat_completion streaming."""

    def handle_completion(self, item: Any, state: StreamState) -> bool:
        """Capture usage and finish_reason from the final HF streaming chunk.

        HF chunks (ChatCompletionStreamOutput) expose usage directly on the
        object — they have no `.object` attribute, so the parent class check
        is skipped and we rely on the presence of `.usage` instead.
        """
        usage = getattr(item, "usage", None)
        if usage is None:
            return False

        state.token_usage = usage
        state.close_stream()

        # Capture finish_reason from this final chunk if not already set.
        if state.finish_reason is None:
            try:
                fr = item.choices[0].finish_reason
                if fr:
                    state.finish_reason = str(fr)
            except (AttributeError, IndexError):
                pass

        return True
