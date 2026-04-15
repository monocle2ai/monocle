"""LangGraph stream processor for stream()/astream() outputs."""

from typing import Any

from monocle_apptrace.instrumentation.common.stream_processor import (
    BaseStreamProcessor,
    StreamState,
)


class LangGraphStreamProcessor(BaseStreamProcessor):
    """Stream processor for LangGraph StreamPart-style chunks.

    Handles v2-style chunks like:
    {
        "type": "messages" | "custom" | "updates" | "values" | ...,
        "ns": (),
        "data": ...,
    }
    """

    def handle_event(self, item: Any, state: StreamState) -> bool:
        return False

    def handle_chunk(self, item: Any, state: StreamState) -> bool:
        if not isinstance(item, dict):
            return False

        part_type = item.get("type")
        data = item.get("data")

        if part_type == "messages":
            # data is typically (message_chunk, metadata)
            if isinstance(data, (list, tuple)) and len(data) > 0:
                message = data[0]
                content = self._extract_message_content(message)
                if content:
                    state.add_content(content)
                if len(data) > 1 and isinstance(data[1], dict):
                    usage = data[1].get("token_usage")
                    if isinstance(usage, dict):
                        state.token_usage = {
                            "completion_tokens": usage.get("completion_tokens"),
                            "prompt_tokens": usage.get("prompt_tokens"),
                            "total_tokens": usage.get("total_tokens"),
                        }
            return True

        if part_type == "custom":
            if data is not None:
                state.add_content(str(data))
            return True

        if part_type in ["updates", "values", "tasks", "checkpoints", "debug"]:
            text = self._extract_text_from_data(data)
            if text:
                # updates/values modes are state snapshots; keep the latest assistant text
                state.update_first_token_time()
                state.accumulated_response = text
            return True

        # Fallback for common LangGraph stream snapshot shapes:
        # - values mode: {"messages": [...]}
        # - updates mode: {"agent": {"messages": [...]}, ...}
        text = self._extract_text_from_stream_item(item)
        if text:
            state.update_first_token_time()
            state.accumulated_response = text
            usage = self._extract_usage_from_stream_item(item)
            if usage is not None:
                state.token_usage = usage
            return True

        return False

    def handle_completion(self, item: Any, state: StreamState) -> bool:
        if not isinstance(item, dict):
            return False

        data = item.get("data")
        usage = self._extract_usage_from_data(data)
        if usage is not None:
            state.token_usage = usage
            return True
        return False

    def _extract_message_content(self, message: Any) -> str:
        content = getattr(message, "content", None)
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict):
                    text = part.get("text") or part.get("content")
                    if text:
                        parts.append(str(text))
            return "".join(parts)
        return str(content)

    def _extract_text_from_data(self, data: Any) -> str:
        if data is None:
            return ""
        if isinstance(data, str):
            return data

        # Common LangGraph state update format where message list is included.
        if isinstance(data, dict):
            messages = data.get("messages")
            if isinstance(messages, list) and len(messages) > 0:
                return self._extract_latest_message_text(messages)
            return ""

        return ""

    def _extract_latest_message_text(self, messages: Any) -> str:
        if not isinstance(messages, list) or len(messages) == 0:
            return ""

        # Walk backwards and pick the last non-empty message content.
        for msg in reversed(messages):
            content = getattr(msg, "content", None)
            if content is None and isinstance(msg, dict):
                content = msg.get("content")

            if content is None:
                continue

            if isinstance(content, str):
                if content.strip():
                    return content
                continue

            if isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, str) and part:
                        parts.append(part)
                    elif isinstance(part, dict):
                        text = part.get("text") or part.get("content")
                        if text:
                            parts.append(str(text))
                combined = "".join(parts)
                if combined.strip():
                    return combined
                continue

            text = str(content)
            if text.strip():
                return text

        return ""

    def _extract_text_from_stream_item(self, item: Any) -> str:
        if not isinstance(item, dict):
            return ""

        # values mode shape
        messages = item.get("messages")
        if isinstance(messages, list):
            text = self._extract_latest_message_text(messages)
            if text:
                return text

        # updates mode shape where each node can hold messages
        for value in item.values():
            if isinstance(value, dict):
                nested_messages = value.get("messages")
                if isinstance(nested_messages, list):
                    text = self._extract_latest_message_text(nested_messages)
                    if text:
                        return text

        return ""

    def _extract_usage_from_data(self, data: Any):
        if not isinstance(data, dict):
            return None

        messages = data.get("messages")
        if not isinstance(messages, list) or len(messages) == 0:
            return None

        last = messages[-1]
        metadata = getattr(last, "response_metadata", None)
        if not isinstance(metadata, dict):
            return None

        usage = metadata.get("token_usage")
        if not isinstance(usage, dict):
            return None

        return {
            "completion_tokens": usage.get("completion_tokens"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }

    def _extract_usage_from_stream_item(self, item: Any):
        if not isinstance(item, dict):
            return None

        usage = self._extract_usage_from_data(item)
        if usage is not None:
            return usage

        for value in item.values():
            if isinstance(value, dict):
                usage = self._extract_usage_from_data(value)
                if usage is not None:
                    return usage

        return None
