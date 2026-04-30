from monocle_apptrace.instrumentation.common.constants import SPAN_START_TIME, SPAN_END_TIME


class ReplayHandler:
    """Dummy method bodies that the Monocle wrapper turns into spans.

    Each method exists as an anchor for ``task_wrapper`` plus an output
    processor; the actual span timestamps and attributes are extracted from
    kwargs by the wrapper machinery.
    """

    def handle_turn(self, prompt: str, response: str, **kwargs) -> str:
        """agentic.turn span — outer container for one user→assistant exchange."""
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
            tool_calls=kwargs.get("tool_calls") or [],
            subagents=kwargs.get("subagents") or [],
            inference_rounds=kwargs.get("inference_rounds") or [],
            model=kwargs.get("model", "codex"),
            tokens=kwargs.get("tokens") or {},
            from_agent="Codex CLI",
            _turn_start=turn_start,
            _turn_end=turn_end,
            **timing,
        )
        return response

    def handle_invocation(self, prompt: str, response: str, **kwargs) -> str:
        """agentic.invocation span — Codex's reasoning + tool-use loop."""
        tool_calls = kwargs.get("tool_calls") or []
        subagents = kwargs.get("subagents") or []
        inference_rounds = kwargs.get("inference_rounds") or []
        from_agent = kwargs.get("from_agent", "Codex CLI")

        for ir in inference_rounds:
            output_text = ir.get("output_text", "")
            if not output_text and not ir.get("tokens"):
                continue  # nothing to show — skip empty rounds
            self.handle_inference_round(
                input_text=ir.get("input_text", prompt),
                output_text=output_text,
                model=ir.get("model", kwargs.get("model", "codex")),
                tokens=ir.get("tokens", {}),
                tool_name=ir.get("tool_name", ""),
                finish_reason=ir.get("finish_reason", ""),
                finish_type=ir.get("finish_type", ""),
                **{SPAN_START_TIME: ir.get(SPAN_START_TIME), SPAN_END_TIME: ir.get(SPAN_END_TIME)},
            )

        for tc in tool_calls:
            self._dispatch_tool(tc, from_agent)

        for sa in subagents:
            self.handle_subagent(
                prompt=sa.get("prompt", ""),
                response=sa.get("response", ""),
                agent_role=sa.get("agent_role", "agent"),
                agent_nickname=sa.get("agent_nickname", ""),
                thread_id=sa.get("thread_id", ""),
                tool_calls=sa.get("tool_calls", []),
                tokens=sa.get("tokens", {}),
                model=sa.get("model", "codex"),
                **{SPAN_START_TIME: sa.get(SPAN_START_TIME), SPAN_END_TIME: sa.get(SPAN_END_TIME)},
            )

        return response

    def _dispatch_tool(self, tc, from_agent):
        tool_name = tc["tool_name"]
        kwargs = {
            "tool_name": tool_name,
            "tool_input": tc["tool_input"],
            "tool_output": tc["tool_output"],
            "call_id": tc.get("call_id", ""),
            "from_agent": from_agent,
            SPAN_START_TIME: tc.get(SPAN_START_TIME),
            SPAN_END_TIME: tc.get(SPAN_END_TIME),
        }
        if tool_name.startswith("mcp__"):
            self.handle_mcp_call(**kwargs)
        else:
            self.handle_tool_call(**kwargs)

    def handle_inference_round(self, input_text: str = "", output_text: str = "", **kwargs) -> str:
        """inference span — one LLM call (thinking between tool uses)."""
        return output_text

    def handle_tool_call(self, tool_name: str, tool_input, tool_output, **kwargs):
        """agentic.tool.invocation span — one tool execution."""
        return tool_output

    def handle_mcp_call(self, tool_name: str, tool_input, tool_output, **kwargs):
        """agentic.mcp.invocation span — one MCP tool execution."""
        return tool_output

    def handle_subagent(self, prompt: str = "", response: str = "", **kwargs) -> str:
        """agentic.invocation span for a spawned subagent."""
        nickname = kwargs.get("agent_nickname") or kwargs.get("agent_role", "agent")
        for tc in kwargs.get("tool_calls") or []:
            self._dispatch_tool(tc, from_agent=nickname)
        if response:
            self.handle_inference_round(
                input_text=prompt,
                output_text=response,
                model=kwargs.get("model", "codex"),
                tokens=kwargs.get("tokens", {}),
                finish_reason="end_turn",
                finish_type="success",
                **{SPAN_START_TIME: kwargs.get(SPAN_START_TIME), SPAN_END_TIME: kwargs.get(SPAN_END_TIME)},
            )
        return response
