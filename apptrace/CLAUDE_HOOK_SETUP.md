# Claude Code Hook Setup Guide

This guide explains how to set up Monocle tracing for Claude Code CLI sessions.

## Overview

Monocle instruments Claude Code by registering hooks for all session events. Each hook fires `python -m monocle_apptrace.instrumentation.metamodel.claude_cli`, which records the event and replays it as OpenTelemetry spans at the end of each turn.

## Setup (3 steps)

### 1. Install the Monocle Package

```bash
pip install monocle_apptrace
```

Or from source:
```bash
cd monocle/apptrace && pip install -e .
```

### 2. Set Environment Variables

Add these to your `~/.zshrc` or `~/.bashrc` and reload:

```bash
export OKAHU_API_KEY="your-api-key"
export OKAHU_INGESTION_ENDPOINT="https://ingest.okahu.co/api/v1/trace/ingest"
export MONOCLE_EXPORTER="okahu,file"        # okahu | file | console (combinable)
export MONOCLE_WORKFLOW_NAME="claude-cli"   # labels your traces
```

```bash
source ~/.zshrc
```

### 3. Register the Hooks

```bash
python -m monocle_apptrace claude-install
```

This writes all 11 hooks into `~/.claude/settings.json` non-destructively — existing hooks from other tools are preserved. Re-running is safe (idempotent).

**That's it.** Start a new Claude Code session and traces flow automatically.

---

## How It Works

```
Claude Code session event fires (any of 11 hooks)
           │
           ▼
python -m monocle_apptrace.instrumentation.metamodel.claude_cli
           │  (reads event JSON from stdin)
           ▼
Event Handler — records event to per-session JSONL log
           │
           │  (on Stop event only)
           ▼
Replay Session — reconstructs turn from all events, emits OTel spans
           │
           ▼
Exporters — Okahu / file / console
```

**Hooks registered** (all 11 Claude Code hook events):
`SessionStart`, `UserPromptSubmit`, `PreToolUse`, `PostToolUse`,
`SubagentStart`, `SubagentStop`, `Stop`, `StopFailure`,
`PreCompact`, `PostCompact`, `SessionEnd`

---

## Testing

### Verify with a mock event

```bash
echo '{"session_id":"test","hook_event_name":"Stop","last_assistant_message":"hello"}' \
  | python -m monocle_apptrace.instrumentation.metamodel.claude_cli
```

### Watch live during a real session

```bash
export MONOCLE_EXPORTER=console
claude   # start from this shell so the env var is inherited
```

Spans appear in Claude Code's status bar and in your terminal as each turn completes.

---

## Troubleshooting

**Hooks not running:**
```bash
cat ~/.claude/settings.json   # verify hooks are registered
python -m monocle_apptrace claude-install   # re-run if missing
```

**No traces:**
```bash
env | grep -E "MONOCLE|OKAHU"   # verify env vars are set
export MONOCLE_EXPORTER=console && claude   # watch traces in terminal
```

**Import errors:**
```bash
python -c "import monocle_apptrace; print(monocle_apptrace.__file__)"
pip install -e monocle/apptrace --force-reinstall
```

---

## Files

| Path | Purpose |
|---|---|
| `~/.claude/settings.json` | Claude Code config — hooks registered here by `claude-install` |
| `.monocle/.claude_sessions/` | Per-session event logs (auto-cleaned on `SessionEnd`) |
