"""LangGraph streaming processor."""

from typing import Any

from monocle_apptrace.instrumentation.common.stream_processor import (
    BaseStreamProcessor,
    StreamState,
)


class LanggraphStreamProcessor(BaseStreamProcessor):
    """Process LangGraph streaming chunks into a unified span result.

    Supports values mode (state dict), updates mode (node-keyed dict),
    messages mode (v2 dict or object-based StreamPart), and custom/debug events.
    """

    def handle_event(self, item: Any, state: StreamState) -> bool:
        """Handle object-based StreamParts (older LangGraph / messages mode with attribute access)."""
        item_type = getattr(item, "type", None)
        if item_type is None or isinstance(item, dict):
            return False 

        if item_type != "messages":
            return True  # consumed non-messages event, no content

        data = getattr(item, "data", None)
        if data is None:
            return True

        if isinstance(data, (tuple, list)) and len(data) > 0:
            msg = data[0]
            content = self._extract_message_content(msg)
            if content:
                state.add_content(content)
            self._extract_usage_from_message(msg, state)
            if len(data) > 1 and isinstance(data[1], dict):
                usage = data[1].get("token_usage")
                if isinstance(usage, dict):
                    state.token_usage = {
                        "completion_tokens": usage.get("completion_tokens"),
                        "prompt_tokens": usage.get("prompt_tokens"),
                        "total_tokens": usage.get("total_tokens"),
                    }
        return True

    def handle_chunk(self, item: Any, state: StreamState) -> bool:
        """Handle stream chunks and extract text and token usage."""
        # LangGraph stream tuple: 3-tuple (keys, mode, data) or 2-tuple (mode, data)
        if isinstance(item, tuple) and len(item) in (2, 3) and isinstance(item[-1], dict):
            mode, data = item[-2], item[-1]
            if mode in ("values", "updates") and isinstance(data, dict):
                text = self._extract_text_from_data(data)
                if text:
                    state.update_first_token_time()
                    state.accumulated_response = text
                usage = self._extract_usage_from_data(data)
                if usage:
                    state.token_usage = usage
            return True

        if not isinstance(item, dict):
            return False

        part_type = item.get("type")
        data = item.get("data")

        if part_type == "messages":
            if isinstance(data, (list, tuple)) and len(data) > 0:
                msg = data[0]
                content = self._extract_message_content(msg)
                if content:
                    state.add_content(content)
                self._extract_usage_from_message(msg, state)
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

        if part_type in ("updates", "values", "tasks", "checkpoints", "debug"):
            text = self._extract_text_from_data(data)
            if text:
                state.update_first_token_time()
                state.accumulated_response = text
            usage = self._extract_usage_from_data(data)
            if usage:
                state.token_usage = usage
            return True

        for node_state in item.values():
            if isinstance(node_state, dict) and any(
                isinstance(k, str) and k.endswith("messages") for k in node_state
            ):
                text = self._extract_text_from_data(node_state)
                if text:
                    state.update_first_token_time()
                    state.accumulated_response = text
                usage = self._extract_usage_from_data(node_state)
                if usage:
                    state.token_usage = usage
                return True

        if any(isinstance(k, str) and k.endswith("messages") for k in item):
            text = self._extract_text_from_data(item)
            if text:
                state.update_first_token_time()
                state.accumulated_response = text
            usage = self._extract_usage_from_data(item)
            if usage:
                state.token_usage = usage
            return True

        return False

    def handle_completion(self, item: Any, state: StreamState) -> bool:
        """LangGraph does not emit a separate completion chunk; unused."""
        return False

    def _extract_message_content(self, message: Any) -> str:
        """Extract text from a LangChain message, handling str and list content."""
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

    def _extract_latest_message_text(self, messages: Any) -> str:
        """Return the last non-empty assistant message text from a messages list."""
        if not isinstance(messages, list) or not messages:
            return ""
        for msg in reversed(messages):
            content = getattr(msg, "content", None)
            if content is None and isinstance(msg, dict):
                content = msg.get("content")
            if not content:
                continue
            text = self._extract_message_content(msg) if not isinstance(content, str) else content
            if text.strip():
                return text
        return ""

    def _message_channels(self, data: dict) -> Any:
        """Yield message-list channels from a state dict (allows custom '*messages' keys)."""
        if isinstance(data.get("messages"), list):
            yield data["messages"]
        for k, v in data.items():
            if k != "messages" and isinstance(k, str) and k.endswith("messages") and isinstance(v, list):
                yield v

    def _extract_text_from_data(self, data: Any) -> str:
        """Extract text from a LangGraph state data object."""
        if data is None:
            return ""
        if isinstance(data, str):
            return data
        if isinstance(data, dict):
            for messages in self._message_channels(data):
                text = self._extract_latest_message_text(messages)
                if text:
                    return text
            # node-keyed updates ({node: {state}}): descend one level
            for v in data.values():
                if isinstance(v, dict):
                    for messages in self._message_channels(v):
                        text = self._extract_latest_message_text(messages)
                        if text:
                            return text
        return ""

    def _extract_usage_from_data(self, data: Any) -> dict:
        """Extract token usage from a LangGraph state data dict."""
        if not isinstance(data, dict):
            return {}
        messages = None
        for channel in self._message_channels(data):
            if channel:
                messages = channel
                break
        if not isinstance(messages, list) or not messages:
            return {}
        last = messages[-1]
        meta = getattr(last, "response_metadata", None)
        if not isinstance(meta, dict):
            return {}
        usage = meta.get("token_usage") or meta.get("usage")
        if not isinstance(usage, dict):
            return {}
        return {
            "completion_tokens": usage.get("completion_tokens", usage.get("output_tokens", 0)),
            "prompt_tokens": usage.get("prompt_tokens", usage.get("input_tokens", 0)),
            "total_tokens": usage.get("total_tokens", 0),
        }

    def _extract_usage_from_message(self, msg: Any, state: StreamState) -> None:
        """Pull token-usage from a LangChain message's response_metadata."""
        try:
            meta = getattr(msg, "response_metadata", None)
            if not isinstance(meta, dict):
                return
            usage = meta.get("token_usage") or meta.get("usage")
            if not usage:
                return
            if isinstance(usage, dict):
                state.token_usage = {
                    "completion_tokens": usage.get("completion_tokens", usage.get("output_tokens", 0)),
                    "prompt_tokens": usage.get("prompt_tokens", usage.get("input_tokens", 0)),
                    "total_tokens": usage.get("total_tokens", 0),
                }
            else:
                completion = getattr(usage, "completion_tokens", getattr(usage, "output_tokens", 0))
                prompt = getattr(usage, "prompt_tokens", getattr(usage, "input_tokens", 0))
                state.token_usage = {
                    "completion_tokens": completion,
                    "prompt_tokens": prompt,
                    "total_tokens": getattr(usage, "total_tokens", completion + prompt),
                }
        except Exception:
            pass
