"""Azure AI Inference-specific stream processor."""

from typing import Any

from monocle_apptrace.instrumentation.common.stream_processor import (
    BaseStreamProcessor,
    StreamState,
)


class AzureAIInferenceStreamProcessor(BaseStreamProcessor):
    """Azure AI Inference-specific stream processor."""

    def handle_event(self, item: Any, state: StreamState) -> bool:
        return False

    def handle_chunk(self, item: Any, state: StreamState) -> bool:
        if not (hasattr(item, "choices") and item.choices and
                hasattr(item.choices[0], "delta") and item.choices[0].delta):
            return False

        choice = item.choices[0]
        delta = choice.delta

        if hasattr(delta, "role") and delta.role:
            state.role = delta.role

        if hasattr(delta, "content") and delta.content:
            state.add_content(delta.content)

        return True

    def handle_completion(self, item: Any, state: StreamState) -> bool:
        if not (hasattr(item, "usage") and item.usage):
            return False

        state.token_usage = item.usage
        state.close_stream()
        return True
