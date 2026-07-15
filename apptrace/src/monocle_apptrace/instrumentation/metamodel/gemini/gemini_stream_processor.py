"""Gemini streaming processor for google.genai generate_content_stream."""

from typing import Any

from monocle_apptrace.instrumentation.common.stream_processor import (
    BaseStreamProcessor,
    StreamState,
)


class GeminiStreamProcessor(BaseStreamProcessor):
    """Stream processor for Gemini generate_content_stream responses.

    Gemini streaming yields GenerateContentResponse chunks where:
    - Each chunk has partial text in candidates[0].content.parts[0].text
    - The final chunk carries usage_metadata (candidates_token_count, etc.)
    - Both text and usage_metadata can appear in the same (final) chunk
    """

    def handle_event(self, item: Any, state: StreamState) -> bool:
        # Gemini does not use SSE-style response.* events
        return False

    def handle_chunk(self, item: Any, state: StreamState) -> bool:
        """Extract partial text or function call from a Gemini streaming chunk."""
        found = False

        # Preferred path: candidates[0].content.parts
        if hasattr(item, "candidates") and item.candidates:
            candidate = item.candidates[0]
            if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                for part in candidate.content.parts:
                    if hasattr(part, "text") and part.text:
                        state.add_content(part.text)
                        found = True
                    elif hasattr(part, "function_call") and part.function_call is not None:
                        fc = part.function_call
                        state.tools.append({
                            "name": getattr(fc, "name", ""),
                            "args": getattr(fc, "args", {}),
                        })
                        # Gemini's final chunk always carries finish_reason=STOP even
                        # for function calls, so we must set FUNCTION_CALL explicitly here.
                        state.finish_reason = "FUNCTION_CALL"
                        found = True

        # Fallback: top-level .text property
        if not found and hasattr(item, "text") and item.text:
            state.add_content(item.text)
            found = True

        return found

    def handle_completion(self, item: Any, state: StreamState) -> bool:
        """Capture usage_metadata and finish_reason from the final Gemini chunk."""
        usage = getattr(item, "usage_metadata", None)
        if usage is None:
            return False

        state.token_usage = {
            "completion_tokens": getattr(usage, "candidates_token_count", 0),
            "prompt_tokens": getattr(usage, "prompt_token_count", 0),
            "total_tokens": getattr(usage, "total_token_count", 0),
        }
        # Capture finish_reason from the final chunk if not already set (e.g. by a function call)
        if state.finish_reason is None:
            try:
                fr = item.candidates[0].finish_reason
                if fr is not None:
                    # Use .name to get the clean enum key (e.g. "STOP") that
                    # matches GEMINI_FINISH_REASON_MAPPING, falling back to str()
                    state.finish_reason = getattr(fr, "name", str(fr))
            except (AttributeError, IndexError, TypeError):
                pass
        state.close_stream()
        return True

    def try_framework_specific_processing(self, item: Any, state: StreamState) -> bool:
        """Override to allow both text extraction AND usage capture on the same chunk."""
        has_chunk = self.handle_chunk(item, state)
        has_completion = self.handle_completion(item, state)
        return has_chunk or has_completion
