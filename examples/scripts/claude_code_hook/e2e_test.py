#!/usr/bin/env python3
"""
End-to-end tests for Claude Code transcript processor.

Emits spans to Okahu and local .monocle/ file exporter, then prints
trace IDs for MCP verification.

Usage:
    source .env && python examples/scripts/claude_code_hook/e2e_test.py
"""

import json
import os
import sys
import time
import uuid
from pathlib import Path

# Ensure monocle_apptrace is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "apptrace" / "src"))

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.resources import Resource, SERVICE_NAME as SVC_NAME

from monocle_apptrace.exporters.monocle_exporters import get_monocle_exporter
from monocle_apptrace.instrumentation.metamodel.claude_code._helper import build_turns
from monocle_apptrace.instrumentation.metamodel.claude_code.claude_code_processor import (
    process_transcript,
)

SERVICE_NAME_VAL = "claude-cli"
SDK_VERSION = "0.7.6"

try:
    import importlib.metadata
    SDK_VERSION = importlib.metadata.version("monocle_apptrace")
except Exception:
    pass


def _user_msg(text):
    return {
        "type": "user",
        "message": {"role": "user", "content": [{"type": "text", "text": text}]},
        "timestamp": "2026-04-01T17:00:00Z",
    }


def _assistant_msg(text, model="claude-sonnet-4-20250514", msg_id=None,
                   input_tokens=200, output_tokens=60,
                   cache_read=150, cache_creation=20, tool_uses=None):
    if msg_id is None:
        msg_id = f"msg_{uuid.uuid4().hex[:8]}"
    content = []
    if tool_uses:
        content.extend(tool_uses)
    content.append({"type": "text", "text": text})
    return {
        "type": "assistant",
        "message": {
            "id": msg_id,
            "role": "assistant",
            "model": model,
            "content": content,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_read_input_tokens": cache_read,
                "cache_creation_input_tokens": cache_creation,
            },
        },
        "timestamp": "2026-04-01T17:00:01Z",
    }


def _tool_result(tool_use_id, content):
    return {
        "type": "user",
        "message": {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}
            ],
        },
    }


def run_test(test_name, messages, session_id):
    """Run a single test: build turns, emit spans, return trace ID."""
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"{'='*60}")

    # Create fresh TracerProvider for each test (separate trace)
    resource = Resource.create({SVC_NAME: SERVICE_NAME_VAL})
    provider = TracerProvider(resource=resource)
    exporters = get_monocle_exporter()
    print(f"  Exporters: {[type(e).__name__ for e in exporters]}")
    for exp in exporters:
        provider.add_span_processor(SimpleSpanProcessor(exp))

    tracer = provider.get_tracer("monocle.claude-code", "1.0.0")

    turns = build_turns(messages)
    print(f"  Turns built: {len(turns)}")

    emitted = process_transcript(
        session_id=session_id,
        turns=turns,
        tracer=tracer,
        sdk_version=SDK_VERSION,
        service_name=SERVICE_NAME_VAL,
    )
    print(f"  Spans emitted for {emitted} turns")

    # Flush and shutdown
    provider.force_flush(timeout_millis=30000)
    provider.shutdown()

    # The trace ID is set by OTel internally. We need to find it from the
    # file exporter output in .monocle/ folder.
    print(f"  Session: {session_id}")
    print(f"  DONE - spans flushed to Okahu + .monocle/")

    return emitted


def main():
    # Verify env vars
    for var in ["OKAHU_API_KEY", "OKAHU_INGESTION_ENDPOINT", "MONOCLE_EXPORTER"]:
        val = os.environ.get(var)
        if not val:
            print(f"ERROR: {var} not set. Source .env first.")
            return 1
        print(f"  {var}={'***' if 'KEY' in var else val}")

    print(f"\nSDK version: {SDK_VERSION}")
    print(f"Service name: {SERVICE_NAME_VAL}")

    # Record existing .monocle files before tests
    monocle_dir = Path(".monocle")
    monocle_dir.mkdir(exist_ok=True)
    existing_files = set(monocle_dir.glob("monocle_trace_*.json"))

    # =========================================================================
    # Test 1: Prompt/Response ("hi")
    # =========================================================================
    session_1 = f"e2e-test1-{uuid.uuid4().hex[:8]}"
    msgs_1 = [
        _user_msg("hi"),
        _assistant_msg(
            "Hello! How can I help you today?",
            input_tokens=50,
            output_tokens=12,
            cache_read=40,
            cache_creation=5,
        ),
    ]
    run_test("Test 1: Prompt/Response (hi)", msgs_1, session_1)

    # =========================================================================
    # Test 2: Bash tool call ("Run ls here")
    # =========================================================================
    session_2 = f"e2e-test2-{uuid.uuid4().hex[:8]}"
    bash_tool = {
        "type": "tool_use",
        "id": "toolu_bash_e2e_001",
        "name": "Bash",
        "input": {"command": "ls", "description": "List files in current directory"},
    }
    msgs_2 = [
        _user_msg("Run ls here in this directory"),
        _assistant_msg(
            "Here are the files in the current directory:",
            tool_uses=[bash_tool],
            input_tokens=180,
            output_tokens=45,
        ),
        _tool_result("toolu_bash_e2e_001",
                      "README.md\napptrace\nexamples\npyproject.toml\nsetup.py"),
    ]
    run_test("Test 2: Bash Tool Call (ls)", msgs_2, session_2)

    # =========================================================================
    # Test 3: Subagent call ("Create a subagent to do 1+1")
    # =========================================================================
    session_3 = f"e2e-test3-{uuid.uuid4().hex[:8]}"
    agent_tool = {
        "type": "tool_use",
        "id": "toolu_agent_e2e_001",
        "name": "Agent",
        "input": {
            "subagent_type": "general-purpose",
            "description": "Calculate 1+1",
            "prompt": "Calculate 1+1 and return the result",
        },
    }
    msgs_3 = [
        _user_msg("Create a subagent to do 1+1 and wait for the agent to return, then add 1 more then return"),
        _assistant_msg(
            "The subagent calculated 1+1 = 2. Adding 1 more: 2 + 1 = 3. The final answer is 3.",
            tool_uses=[agent_tool],
            input_tokens=250,
            output_tokens=80,
        ),
        _tool_result("toolu_agent_e2e_001", "The result of 1+1 is 2."),
    ]
    run_test("Test 3: Subagent Call (1+1+1)", msgs_3, session_3)

    # Wait for file exporter to flush
    time.sleep(2)

    # =========================================================================
    # Find new trace files
    # =========================================================================
    print(f"\n{'='*60}")
    print("LOCAL TRACE FILES (.monocle/)")
    print(f"{'='*60}")
    new_files = sorted(set(monocle_dir.glob("monocle_trace_*.json")) - existing_files)
    trace_ids = []
    for f in new_files:
        print(f"\n  {f.name}")
        try:
            data = json.loads(f.read_text())
            if isinstance(data, list) and data:
                tid = data[0].get("context", {}).get("trace_id")
                if tid:
                    trace_ids.append(tid)
                    print(f"    trace_id: {tid}")
                    # Print span summary
                    for span in data:
                        stype = span.get("attributes", {}).get("span.type", "?")
                        sname = span.get("name", "?")
                        etype = span.get("attributes", {}).get("entity.1.type", "")
                        ename = span.get("attributes", {}).get("entity.1.name", "")
                        print(f"    span: {sname} | type={stype} | entity={etype}:{ename}")
        except Exception as e:
            print(f"    ERROR reading: {e}")

    print(f"\n{'='*60}")
    print("TRACE IDS FOR OKAHU MCP VERIFICATION")
    print(f"{'='*60}")
    for i, tid in enumerate(trace_ids):
        print(f"  Test {i+1}: {tid}")

    print(f"\nUse these trace IDs with Okahu MCP:")
    print(f"  mcp__okahu-mcp__get_trace_spans(workflow_name='claude-cli', trace_id='<id>')")

    return 0


if __name__ == "__main__":
    sys.exit(main())
