from monocle_apptrace.instrumentation.common.constants import SPAN_START_TIME, SPAN_END_TIME


class StopFailureError(Exception):
    """Raised when a turn ends with a StopFailure hook event."""
    def __init__(self, error: str, details: str = ""):
        super().__init__(details or error)
        self.code = error


class ReplayHandler:

    def handle_turn(self, prompt: str, response: str, **kwargs) -> str:
        """agentic.turn span — outer container for one user→assistant exchange."""
        tool_calls = kwargs.get("tool_calls") or []
        subagents = kwargs.get("subagents") or []
        inference_rounds = kwargs.get("inference_rounds") or []
        model = kwargs.get("model", "claude")
        tokens = kwargs.get("tokens") or {}

        # _turn_start/_turn_end carry the ISO timestamps through because ClaudeSpanHandler
        # already popped SPAN_START_TIME/SPAN_END_TIME from kwargs before this body runs.
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
        """agentic.invocation span — Claude's reasoning + tool-use loop."""
        tool_calls = kwargs.get("tool_calls") or []
        subagents = kwargs.get("subagents") or []
        inference_rounds = kwargs.get("inference_rounds") or []

        for ir in inference_rounds:
            output_text = ir.get("output_text", "")
            if not output_text:
                continue  # skip rounds with no visible text output (tool-dispatch-only rounds)
            self.handle_inference_round(
                input_text=ir.get("input_text", prompt),
                output_text=output_text,
                model=ir.get("model", kwargs.get("model", "claude")),
                tokens=ir.get("tokens", {}),
                tool_name=ir.get("tool_name", ""),
                finish_reason=ir.get("finish_reason", ""),
                finish_type=ir.get("finish_type", ""),
                **{SPAN_START_TIME: ir.get(SPAN_START_TIME), SPAN_END_TIME: ir.get(SPAN_END_TIME)},
            )

        for tc in tool_calls:
            tool_name = tc["tool_name"]
            if tool_name.startswith("mcp__"):
                self.handle_mcp_call(
                    tool_name=tool_name,
                    tool_input=tc["tool_input"],
                    tool_output=tc["tool_output"],
                    **{SPAN_START_TIME: tc.get(SPAN_START_TIME), SPAN_END_TIME: tc.get(SPAN_END_TIME)},
                )
            else:
                self.handle_tool_call(
                    tool_name=tool_name,
                    tool_input=tc["tool_input"],
                    tool_output=tc["tool_output"],
                    **{SPAN_START_TIME: tc.get(SPAN_START_TIME), SPAN_END_TIME: tc.get(SPAN_END_TIME)},
                )

        for sa in subagents:
            self.handle_subagent(
                prompt=sa.get("prompt", ""),
                response=sa.get("response", ""),
                agent_id=sa.get("agent_id", ""),
                agent_type=sa.get("agent_type", "agent"),
                description=sa.get("description", ""),
                tool_calls=sa.get("tool_calls", []),
                tokens=sa.get("tokens", {}),
                model=sa.get("model", "claude"),
                **{SPAN_START_TIME: sa.get(SPAN_START_TIME), SPAN_END_TIME: sa.get(SPAN_END_TIME)},
            )

        failure = getattr(self, "_stop_failure", "")
        if failure:
            raise StopFailureError(failure, getattr(self, "_stop_failure_details", ""))

        return response

    def handle_inference_round(self, input_text: str = "", output_text: str = "", **kwargs) -> str:
        """inference span — one LLM call (thinking between tool uses)."""
        return output_text

    def handle_tool_call(self, tool_name: str, tool_input: dict, tool_output: dict, **kwargs) -> dict:
        """agentic.tool.invocation span — one tool execution."""
        return tool_output

    def handle_mcp_call(self, tool_name: str, tool_input: dict, tool_output: dict, **kwargs) -> dict:
        """agentic.mcp.invocation span — one MCP tool execution."""
        return tool_output

    def handle_subagent(self, prompt: str = "", response: str = "", **kwargs) -> str:
        """agentic.invocation span for a spawned subagent."""
        tool_calls = kwargs.get("tool_calls") or []
        for tc in tool_calls:
            tool_name = tc["tool_name"]
            if tool_name.startswith("mcp__"):
                self.handle_mcp_call(
                    tool_name=tool_name,
                    tool_input=tc["tool_input"],
                    tool_output=tc["tool_output"],
                    **{SPAN_START_TIME: tc.get(SPAN_START_TIME), SPAN_END_TIME: tc.get(SPAN_END_TIME)},
                )
            else:
                self.handle_tool_call(
                    tool_name=tool_name,
                    tool_input=tc["tool_input"],
                    tool_output=tc["tool_output"],
                    **{SPAN_START_TIME: tc.get(SPAN_START_TIME), SPAN_END_TIME: tc.get(SPAN_END_TIME)},
                )
        # One inference span per subagent capturing its final response and aggregate tokens
        if response:
            self.handle_inference_round(
                input_text=prompt,
                output_text=response,
                model=kwargs.get("model", "claude"),
                tokens=kwargs.get("tokens", {}),
                finish_reason="end_turn",
                finish_type="success",
                **{SPAN_START_TIME: kwargs.get(SPAN_START_TIME), SPAN_END_TIME: kwargs.get(SPAN_END_TIME)},
            )
        return response

