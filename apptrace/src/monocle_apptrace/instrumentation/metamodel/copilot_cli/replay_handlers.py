from monocle_apptrace.instrumentation.common.constants import SPAN_START_TIME, SPAN_END_TIME

_MCP_BUILTIN_TOOLS = {"read_mcp_resource", "list_mcp_resources", "list_mcp_resource_templates"}


class ToolFailureError(Exception):
    """Raised so task_wrapper marks the span ERROR with this as status.message."""


class InterruptedTurnError(Exception):
    """Raised when an interaction ended without an assistant response (user
    cancelled, Ctrl+C, network drop). Lets the wrapper mark the turn and
    invocation spans as ERROR without aborting the rest of the replay."""


class ReplayHandler:
    """Empty method bodies — task_wrapper turns each call into a span via methods.py."""

    def handle_turn(self, prompt: str, response: str, **kwargs) -> str:
        self.handle_invocation(
            prompt=prompt,
            response=response,
            tool_calls=kwargs.get("tool_calls") or [],
            inference_rounds=kwargs.get("inference_rounds") or [],
            model=kwargs.get("model", "copilot"),
            interrupted=bool(kwargs.get("interrupted")),
            from_agent="Copilot CLI",
            **{SPAN_START_TIME: kwargs.get("_turn_start"),
               SPAN_END_TIME: kwargs.get("_turn_end")},
        )
        return response

    def handle_invocation(self, prompt: str, response: str, **kwargs) -> str:
        tool_calls = kwargs.get("tool_calls") or []
        inference_rounds = kwargs.get("inference_rounds") or []
        from_agent = kwargs.get("from_agent", "Copilot CLI")

        for ir in inference_rounds:
            if not ir.get("output_text") and not ir.get("tokens"):
                continue
            self.handle_inference_round(
                input_text=ir.get("input_text", prompt),
                output_text=ir.get("output_text", ""),
                model=ir.get("model", kwargs.get("model", "copilot")),
                tokens=ir.get("tokens", {}),
                finish_reason=ir.get("finish_reason", ""),
                finish_type=ir.get("finish_type", ""),
                **{SPAN_START_TIME: ir.get(SPAN_START_TIME), SPAN_END_TIME: ir.get(SPAN_END_TIME)},
            )

        for tc in tool_calls:
            self._dispatch_tool(tc, from_agent)

        # Raise AFTER processing tool calls so any orphan tool spans still emit
        # alongside the ERROR turn/invocation spans.
        if kwargs.get("interrupted"):
            raise InterruptedTurnError("interaction interrupted before assistant responded")

        return response

    def _dispatch_tool(self, tc, from_agent):
        tool_name = tc["tool_name"]
        kwargs = {
            "tool_name": tool_name,
            "tool_input": tc["tool_input"],
            "tool_output": tc["tool_output"],
            "call_id": tc.get("call_id", ""),
            "failed": tc.get("failed", False),
            "error_message": tc.get("error_message", ""),
            "error_code": tc.get("error_code", ""),
            "from_agent": from_agent,
            SPAN_START_TIME: tc.get(SPAN_START_TIME),
            SPAN_END_TIME: tc.get(SPAN_END_TIME),
        }
        try:
            if tool_name.startswith("mcp__") or tool_name in _MCP_BUILTIN_TOOLS:
                self.handle_mcp_call(**kwargs)
            else:
                self.handle_tool_call(**kwargs)
        except ToolFailureError:
            pass

    def handle_inference_round(self, input_text: str = "", output_text: str = "", **kwargs) -> str:
        return output_text

    def handle_tool_call(self, tool_name: str, tool_input, tool_output, **kwargs):
        if kwargs.get("failed"):
            err = kwargs.get("error_message") or str(tool_output) or f"{tool_name} failed"
            code = kwargs.get("error_code") or ""
            raise ToolFailureError(f"[{code}] {err}" if code else err)
        return tool_output

    def handle_mcp_call(self, tool_name: str, tool_input, tool_output, **kwargs):
        if kwargs.get("failed"):
            err = kwargs.get("error_message") or str(tool_output) or f"{tool_name} failed"
            code = kwargs.get("error_code") or ""
            raise ToolFailureError(f"[{code}] {err}" if code else err)
        return tool_output

    def handle_session_summary(self, totals: dict = None, **kwargs):
        """Session-scoped span carrying the session.shutdown rollup."""
        return totals or {}
