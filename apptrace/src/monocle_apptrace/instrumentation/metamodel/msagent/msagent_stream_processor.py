"""MS Agent Framework streaming processor for inference responses."""

import json
from typing import Any

from monocle_apptrace.instrumentation.common.stream_processor import (
    BaseStreamProcessor,
    StreamEventTypes,
    StreamState,
)


class MSAgentStreamProcessor(BaseStreamProcessor):
    """Stream processor for Microsoft Agent Framework inference streams."""

    @staticmethod
    def _field(obj: Any, key: str, default: Any = None) -> Any:
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    @staticmethod
    def _as_list(value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        return [value]

    def _extract_text_from_message_parts(self, content_parts: Any) -> str:
        """Extract text from message content parts in object or dict form."""
        text_parts: list[str] = []
        for part in self._as_list(content_parts):
            part_type = self._field(part, "type")
            if part_type in ("output_text", "text"):
                text = self._field(part, "text", "")
                # Some SDK payloads wrap text in an object: {"text": {"value": "..."}}
                if isinstance(text, dict):
                    text = text.get("value", "")
                if not text:
                    text = self._field(part, "value", "")
                if text:
                    text_parts.append(str(text))
        return "".join(text_parts)

    def _normalize_item(self, item: Any) -> Any:
        """Normalize SSE/event-wrapper items into dict/object payloads.

        Some SDKs emit objects with `event` + JSON `data` instead of direct
        `type` payloads. This converts those into a structure expected by
        the existing response.* handlers.
        """
        if item is None:
            return item

        event_name = self._field(item, "event")
        data_payload = self._field(item, "data")

        # Tuple/list stream form: (event_name, data)
        if not event_name and isinstance(item, (tuple, list)) and len(item) >= 2:
            if isinstance(item[0], str):
                event_name = item[0]
                data_payload = item[1]

        # Raw SSE line block form in string/bytes.
        if not event_name and isinstance(item, (str, bytes)):
            raw_text = item.decode("utf-8", errors="ignore") if isinstance(item, bytes) else item
            if "event:" in raw_text and "data:" in raw_text:
                parsed_event = ""
                parsed_data: Any = ""
                for line in raw_text.splitlines():
                    line = line.strip()
                    if line.startswith("event:"):
                        parsed_event = line.split(":", 1)[1].strip()
                    elif line.startswith("data:"):
                        parsed_data = line.split(":", 1)[1].strip()
                if parsed_event:
                    event_name = parsed_event
                    data_payload = parsed_data

        if not event_name:
            return item

        parsed: Any = {}
        if isinstance(data_payload, str) and data_payload.strip():
            try:
                parsed = json.loads(data_payload)
            except Exception:
                parsed = {"data": data_payload}
        elif isinstance(data_payload, dict):
            parsed = dict(data_payload)
        else:
            parsed = {}

        if isinstance(parsed, dict) and "type" not in parsed:
            parsed["type"] = str(event_name)
        return parsed

    def _add_tool(self, tool_id: str, tool_name: str, tool_arguments: str, state: StreamState) -> None:
        if not (tool_id or tool_name or tool_arguments):
            return
        state.tools.append(
            {
                "id": tool_id or "",
                "name": tool_name or "",
                "arguments": tool_arguments or "",
            }
        )
        if state.finish_reason is None:
            state.finish_reason = "tool_calls"

    def _capture_tool_item(self, output_item: Any, state: StreamState) -> None:
        """Capture function call data from response output items."""
        item_type = self._field(output_item, "type")
        if item_type == "function_call":
            tool_id = self._field(output_item, "id", "") or self._field(output_item, "call_id", "")
            tool_name = self._field(output_item, "name", "")
            tool_arguments = self._field(output_item, "arguments", "")
            self._add_tool(str(tool_id), str(tool_name), str(tool_arguments), state)

        # Function-call outputs can help avoid empty output in tool-call-only turns.
        if item_type == "function_call_output":
            tool_output = self._field(output_item, "output", "")
            if tool_output and not state.accumulated_response:
                state.add_content(str(tool_output))

    def _capture_completed_response(self, response_obj: Any, state: StreamState) -> None:
        """Capture usage and output content from response.completed payload."""
        usage = self._field(response_obj, "usage")
        if usage:
            state.token_usage = usage

        for output_item in self._field(response_obj, "output", []) or []:
            self._capture_tool_item(output_item, state)

            # Capture assistant message content if present.
            if self._field(output_item, "type") == "message":
                text = self._extract_text_from_message_parts(self._field(output_item, "content", []))
                if text:
                    state.add_content(text)

        # Some responses provide a top-level output_text convenience field.
        if not state.accumulated_response:
            output_text = self._field(response_obj, "output_text", "")
            if output_text:
                state.add_content(str(output_text))

    def _reconstruct_output_from_raw_items(self, state: StreamState) -> None:
        """Fallback pass: recover assistant output from raw items if live parsing missed it."""
        recovered_text_parts: list[str] = []
        recovered_tool_output: str = ""

        for raw_item in state.raw_items:
            item = self._normalize_item(raw_item)

            # Direct text-bearing items.
            direct_text = self._field(item, "text", "")
            if direct_text:
                recovered_text_parts.append(str(direct_text))

            event_type = self._field(item, "type")
            if event_type in (StreamEventTypes.RESPONSE_OUTPUT_TEXT_DELTA, StreamEventTypes.RESPONSE_TEXT_DELTA):
                delta = self._field(item, "delta", "")
                if delta:
                    recovered_text_parts.append(str(delta))

            if event_type in ("response.output_item.added", "response.output_item.done"):
                output_item = self._field(item, "item")
                item_type = self._field(output_item, "type")
                if item_type == "message":
                    text = self._extract_text_from_message_parts(self._field(output_item, "content", []))
                    if text:
                        recovered_text_parts.append(text)
                elif item_type == "function_call":
                    self._add_tool(
                        str(self._field(output_item, "id", "") or self._field(output_item, "call_id", "")),
                        str(self._field(output_item, "name", "") or ""),
                        str(self._field(output_item, "arguments", "") or ""),
                        state,
                    )
                elif item_type == "function_call_output" and not recovered_tool_output:
                    recovered_tool_output = str(self._field(output_item, "output", "") or "")

            if event_type == StreamEventTypes.RESPONSE_COMPLETED:
                response_obj = self._field(item, "response")
                if response_obj is not None:
                    for output_item in self._as_list(self._field(response_obj, "output", [])):
                        item_type = self._field(output_item, "type")
                        if item_type == "message":
                            text = self._extract_text_from_message_parts(self._field(output_item, "content", []))
                            if text:
                                recovered_text_parts.append(text)
                        elif item_type == "function_call":
                            self._add_tool(
                                str(self._field(output_item, "id", "") or self._field(output_item, "call_id", "")),
                                str(self._field(output_item, "name", "") or ""),
                                str(self._field(output_item, "arguments", "") or ""),
                                state,
                            )
                        elif item_type == "function_call_output" and not recovered_tool_output:
                            recovered_tool_output = str(self._field(output_item, "output", "") or "")

                    completed_output_text = self._field(response_obj, "output_text", "")
                    if completed_output_text:
                        recovered_text_parts.append(str(completed_output_text))

        reconstructed_text = "".join(recovered_text_parts).strip()
        if reconstructed_text:
            state.accumulated_response = reconstructed_text
        elif recovered_tool_output:
            state.accumulated_response = recovered_tool_output

    def _build_tool_summary_text(self, state: StreamState) -> str:
        """Build a compact synthetic summary for tool-call-only turns."""
        if not state.tools:
            return ""

        tool_summaries: list[str] = []
        for tool in state.tools:
            name = str(tool.get("name", "") or "unknown_tool")
            arguments = str(tool.get("arguments", "") or "")
            if arguments:
                tool_summaries.append(f"{name}({arguments})")
            else:
                tool_summaries.append(name)

        return "Tool calls: " + "; ".join(tool_summaries)

    def _has_tool_call_signal(self, state: StreamState) -> bool:
        """Best-effort detection of tool-call turns when details are sparse."""
        if state.tools:
            return True
        if state.finish_reason in ("tool_calls", "function_call"):
            return True

        for raw_item in state.raw_items:
            item = self._normalize_item(raw_item)
            item_type = self._field(item, "type")
            if item_type in (
                "response.function_call.delta",
                "response.function_call_arguments.done",
            ):
                return True

            if item_type in ("response.output_item.added", "response.output_item.done"):
                output_item = self._field(item, "item")
                if self._field(output_item, "type") in ("function_call", "function_call_output"):
                    return True

            response_obj = self._field(item, "response")
            for output_item in self._as_list(self._field(response_obj, "output", [])):
                if self._field(output_item, "type") in ("function_call", "function_call_output"):
                    return True

        return False

    def handle_event(self, item: Any, state: StreamState) -> bool:
        """Handle response.* style events emitted by some OpenAI-compatible streams."""
        normalized_item = self._normalize_item(item)
        event_type = self._field(normalized_item, "type")
        if not StreamEventTypes.is_response_event(str(event_type) if event_type is not None else ""):
            return False

        state.update_first_token_time()

        if event_type in (StreamEventTypes.RESPONSE_OUTPUT_TEXT_DELTA, StreamEventTypes.RESPONSE_TEXT_DELTA):
            delta_text = self._field(normalized_item, "delta")
            if delta_text:
                state.add_content(str(delta_text))
        elif event_type == StreamEventTypes.RESPONSE_FUNCTION_CALL_DELTA:
            delta = self._field(normalized_item, "delta", {})
            tool_name = self._field(delta, "name", "") or self._field(normalized_item, "name", "")
            tool_arguments = self._field(delta, "arguments", "") or self._field(normalized_item, "arguments", "")
            tool_id = self._field(delta, "id", "") or self._field(normalized_item, "id", "")
            self._add_tool(str(tool_id), str(tool_name), str(tool_arguments), state)
        elif event_type == "response.function_call_arguments.done":
            tool_arguments = self._field(normalized_item, "arguments", "")
            if tool_arguments and state.tools:
                state.tools[-1]["arguments"] = str(tool_arguments)
        elif event_type == "response.output_item.added":
            self._capture_tool_item(self._field(normalized_item, "item"), state)
        elif event_type == "response.output_item.done":
            self._capture_tool_item(self._field(normalized_item, "item"), state)
        elif event_type == StreamEventTypes.RESPONSE_TEXT_DONE:
            text = self._field(normalized_item, "text")
            if text:
                state.add_content(str(text))
            state.close_stream()
        elif event_type == StreamEventTypes.RESPONSE_COMPLETED:
            self._capture_completed_response(self._field(normalized_item, "response"), state)
            state.close_stream()
            response_obj = self._field(normalized_item, "response")
            if self._field(response_obj, "usage"):
                state.token_usage = self._field(response_obj, "usage")

        return True

    def handle_chunk(self, item: Any, state: StreamState) -> bool:
        """Handle chunk structures from chat/assistants streaming updates."""
        normalized_item = self._normalize_item(item)
        handled = False

        # Agent framework updates often expose text directly.
        item_text = self._field(normalized_item, "text")
        if item_text:
            state.add_content(str(item_text))
            handled = True

        # OpenAI/ChatCompletion chunk compatibility.
        choices = self._as_list(self._field(normalized_item, "choices"))
        if choices:
            first_choice = choices[0]
            delta = self._field(first_choice, "delta")
            if delta:
                delta_content = self._field(delta, "content")
                if delta_content:
                    state.add_content(str(delta_content))
                    handled = True

                # Some responses emit text fragments in delta.text or delta.output_text.
                delta_text = self._field(delta, "text") or self._field(delta, "output_text")
                if delta_text:
                    state.add_content(str(delta_text))
                    handled = True

            if self._field(delta, "tool_calls"):
                handled = True

        # agent_framework pattern: tool calls and usage arrive via ChatResponseUpdate.contents
        for content in self._as_list(self._field(normalized_item, "contents")):
            content_type = self._field(content, "type")
            if content_type == "function_call":
                tool_id = self._field(content, "call_id", "") or self._field(content, "id", "")
                tool_name = self._field(content, "name", "")
                tool_arguments = self._field(content, "arguments", "")
                self._add_tool(str(tool_id), str(tool_name), str(tool_arguments), state)
                handled = True

        return handled

    def handle_completion(self, item: Any, state: StreamState) -> bool:
        """Capture usage and finish metadata from terminal chunks."""
        normalized_item = self._normalize_item(item)
        handled = False

        usage = self._field(normalized_item, "usage")
        if usage:
            state.token_usage = usage
            handled = True

        # Some frameworks expose usage_details instead of usage.
        usage_details = self._field(normalized_item, "usage_details")
        if usage_details:
            usage = usage_details
            state.token_usage = {
                "completion_tokens": getattr(usage, "output_token_count", 0),
                "prompt_tokens": getattr(usage, "input_token_count", 0),
                "total_tokens": getattr(usage, "total_token_count", 0),
            }
            handled = True

        # agent_framework pattern: UsageContent in ChatResponseUpdate.contents
        for content in self._as_list(self._field(normalized_item, "contents")):
            if self._field(content, "type") == "usage":
                details = self._field(content, "details")
                if details:
                    state.token_usage = {
                        "completion_tokens": getattr(details, "output_token_count", 0),
                        "prompt_tokens": getattr(details, "input_token_count", 0),
                        "total_tokens": getattr(details, "total_token_count", 0),
                    }
                    handled = True

        finish_reason = self._field(normalized_item, "finish_reason")
        if finish_reason:
            state.finish_reason = finish_reason
            handled = True

        choices = self._as_list(self._field(normalized_item, "choices"))
        if choices:
            first_choice = choices[0]
            choice_finish_reason = self._field(first_choice, "finish_reason")
            if choice_finish_reason:
                state.finish_reason = choice_finish_reason
                handled = True

        if handled:
            state.close_stream()

        return handled

    def assemble_data(self, state: StreamState) -> None:
        """Assemble fragmented tool-calls from chunk deltas."""
        tool_fragments: dict[str, dict[str, str]] = {}

        for item in state.raw_items:
            try:
                normalized_item = self._normalize_item(item)
                choices = self._as_list(self._field(normalized_item, "choices"))
                if not choices:
                    continue
                choice = choices[0]

                # Collect finish_reason from completion chunks.
                choice_finish_reason = self._field(choice, "finish_reason")
                if choice_finish_reason:
                    state.finish_reason = str(choice_finish_reason)

                delta = self._field(choice, "delta")
                for tool_call in self._as_list(self._field(delta, "tool_calls")):
                    index = self._field(tool_call, "index")
                    tool_id = self._field(tool_call, "id", "")
                    key = str(index) if index is not None else str(tool_id or len(tool_fragments))
                    fragment = tool_fragments.setdefault(key, {"id": "", "name": "", "arguments": ""})

                    if tool_id:
                        fragment["id"] = str(tool_id)
                    function_obj = self._field(tool_call, "function")
                    function_name = self._field(function_obj, "name") or self._field(tool_call, "name")
                    if function_name:
                        fragment["name"] = str(function_name)
                    function_arguments = self._field(function_obj, "arguments") or self._field(tool_call, "arguments")
                    if function_arguments:
                        fragment["arguments"] += str(function_arguments)

                # Some terminal chunks carry tool calls under message.tool_calls.
                message = self._field(choice, "message")
                for tool_call in self._as_list(self._field(message, "tool_calls")):
                    function_obj = self._field(tool_call, "function")
                    self._add_tool(
                        str(self._field(tool_call, "id", "")),
                        str(self._field(function_obj, "name") or self._field(tool_call, "name") or ""),
                        str(self._field(function_obj, "arguments") or self._field(tool_call, "arguments") or ""),
                        state,
                    )
            except Exception as exc:
                self.logger.warning("Warning: Error while assembling MSAgent stream chunks: %s", str(exc))

        for fragment in tool_fragments.values():
            self._add_tool(fragment["id"], fragment["name"], fragment["arguments"], state)

        if not state.accumulated_response:
            self._reconstruct_output_from_raw_items(state)

        if not state.accumulated_response:
            synthetic_summary = self._build_tool_summary_text(state)
            if synthetic_summary:
                state.accumulated_response = synthetic_summary

        if not state.accumulated_response and self._has_tool_call_signal(state):
            state.accumulated_response = "Tool call requested."

        if not state.accumulated_response:
            state.accumulated_response = "No assistant text captured for this streaming turn."

        # Set a default finish_reason when the stream ends without one.
        if state.finish_reason is None:
            state.finish_reason = "tool_calls" if self._has_tool_call_signal(state) else "stop"