from monocle_apptrace.instrumentation.common.constants import SPAN_START_TIME, SPAN_END_TIME


class ToolFailureError(Exception):
    """Raised so task_wrapper marks the tool span ERROR with this as status.message."""
    def __init__(self, tool_name: str, error: str):
        super().__init__(error or f"Tool '{tool_name}' failed")
        self.tool_name = tool_name


class ReplayHandler:
    """Empty method bodies — methods.py wraps each call into a span via task_wrapper."""

    def handle_turn(self, prompt: str, response: str, **kwargs) -> str:
        tool_calls = kwargs.get("tool_calls") or []
        subagents = kwargs.get("subagents") or []
        inference_rounds = kwargs.get("inference_rounds") or []
        model = kwargs.get("model", "copilot")
        tokens = kwargs.get("tokens") or {}

        turn_start = kwargs.get("_turn_start")
        turn_end = kwargs.get("_turn_end")
        timing = {}
        if turn_start:
            timing[SPAN_START_TIME] = turn_start
        if turn_end:
            timing[SPAN_END_TIME] = turn_end

        self.handle_invocation(
            prompt=prompt,
            response=response,
            tool_calls=tool_calls,
            subagents=subagents,
            inference_rounds=inference_rounds,
            model=model,
            tokens=tokens,
            _turn_start=turn_start,
            _turn_end=turn_end,
            **timing,
        )
        return response

    def handle_invocation(self, prompt: str, response: str, **kwargs) -> str:
        tool_calls = kwargs.get("tool_calls") or []
        subagents = kwargs.get("subagents") or []
        inference_rounds = kwargs.get("inference_rounds") or []

        for ir in inference_rounds:
            output_text = ir.get("output_text", "")
            # In v1 we synthesize one round per turn from hook timestamps with no
            # output_text. Emit it anyway so the inference span is always present —
            # v1.1 (transcript parsing) will fill in the actual model response.
            self.handle_inference_round(
                input_text=ir.get("input_text", prompt),
                output_text=output_text,
                model=ir.get("model", kwargs.get("model", "copilot")),
                tokens=ir.get("tokens", {}),
                tool_name=ir.get("tool_name", ""),
                finish_reason=ir.get("finish_reason", ""),
                finish_type=ir.get("finish_type", ""),
                otel_trace_id=ir.get("otel_trace_id", ""),
                **{SPAN_START_TIME: ir.get(SPAN_START_TIME), SPAN_END_TIME: ir.get(SPAN_END_TIME)},
            )

        for tc in tool_calls:
            self._dispatch_tool(tc)

        for sa in subagents:
            self.handle_subagent(
                prompt=sa.get("prompt", ""),
                response=sa.get("response", ""),
                agent_id=sa.get("agent_id", ""),
                agent_type=sa.get("agent_type", "agent"),
                description=sa.get("description", ""),
                tool_calls=sa.get("tool_calls", []),
                tokens=sa.get("tokens", {}),
                model=sa.get("model", "copilot"),
                **{SPAN_START_TIME: sa.get(SPAN_START_TIME), SPAN_END_TIME: sa.get(SPAN_END_TIME)},
            )

        return response

    def _dispatch_tool(self, tc):
        tool_name = tc["tool_name"]
        extra = {"failed": True, "error": tc.get("error", "")} if tc.get("failed") else {}
        try:
            if tool_name.startswith("mcp__"):
                self.handle_mcp_call(
                    tool_name=tool_name,
                    tool_input=tc["tool_input"],
                    tool_output=tc["tool_output"],
                    **extra,
                    **{SPAN_START_TIME: tc.get(SPAN_START_TIME), SPAN_END_TIME: tc.get(SPAN_END_TIME)},
                )
            else:
                self.handle_tool_call(
                    tool_name=tool_name,
                    tool_input=tc["tool_input"],
                    tool_output=tc["tool_output"],
                    **extra,
                    **{SPAN_START_TIME: tc.get(SPAN_START_TIME), SPAN_END_TIME: tc.get(SPAN_END_TIME)},
                )
        except ToolFailureError:
            pass  # span already recorded with ERROR status

    def handle_inference_round(self, input_text: str = "", output_text: str = "", **kwargs) -> str:
        return output_text

    def handle_tool_call(self, tool_name: str, tool_input: dict, tool_output: dict, **kwargs) -> dict:
        if kwargs.get("failed"):
            error_msg = kwargs.get("error", "") or f"Tool '{tool_name}' failed"
            raise ToolFailureError(tool_name, error_msg)
        return tool_output

    def handle_mcp_call(self, tool_name: str, tool_input: dict, tool_output: dict, **kwargs) -> dict:
        if kwargs.get("failed"):
            error_msg = kwargs.get("error", "") or f"Tool '{tool_name}' failed"
            raise ToolFailureError(tool_name, error_msg)
        return tool_output

    def handle_subagent(self, prompt: str = "", response: str = "", **kwargs) -> str:
        tool_calls = kwargs.get("tool_calls") or []
        for tc in tool_calls:
            self._dispatch_tool(tc)
        if response:
            self.handle_inference_round(
                input_text=prompt,
                output_text=response,
                model=kwargs.get("model", "copilot"),
                tokens=kwargs.get("tokens", {}),
                finish_reason="end_turn",
                finish_type="success",
                **{SPAN_START_TIME: kwargs.get(SPAN_START_TIME), SPAN_END_TIME: kwargs.get(SPAN_END_TIME)},
            )
        return response
