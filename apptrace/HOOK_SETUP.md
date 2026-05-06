# Monocle Hook Setup Guide

This guide covers setting up Monocle tracing for Claude Code and Codex CLI.

## Setup

### 1. Install the Monocle Package

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

uv tool install monocle_apptrace
```

### 2. Register Hooks

**Claude Code:**
```bash
monocle-apptrace claude-setup
```

**Codex CLI:**
```bash
monocle-apptrace codex-setup
```

Both commands prompt for your Okahu API key and save it to `~/.monocle/.env`. Leave blank to export to a local file only.

**Optional overrides** — add to `~/.zshrc` or `~/.bashrc` if you need them:

```bash
export MONOCLE_EXPORTER="okahu,file"        # okahu | file (combinable)
export MONOCLE_WORKFLOW_NAME="my-project"   # defaults to current directory name
```

Start a new session — traces flow automatically.

---

## How It Works

```
Agent session event fires
           │
           ▼
monocle-apptrace {claude,codex}-hook
           │  (reads event JSON from stdin)
           ▼
Event Handler — records event to per-session JSONL log
           │
           │  (on Stop event only)
           ▼
Replay Session — reconstructs turns, emits OTel spans
           │
           ▼
Exporters — Okahu / file / console
```

**Claude Code hooks** (all 11 events):
`SessionStart`, `UserPromptSubmit`, `PreToolUse`, `PostToolUse`,
`SubagentStart`, `SubagentStop`, `Stop`, `StopFailure`,
`PreCompact`, `PostCompact`, `SessionEnd`

**Codex hooks:** `SessionStart`, `SessionStop`

**Spans emitted per turn:**
- `agentic.turn` — the full user turn (prompt → response)
- `inference` — one per LLM inference round
- `agentic.tool.invocation` — one per tool call
- `agentic.mcp.invocation` — one per MCP tool call
- `agentic.invocation` — one per sub-agent spawned

---