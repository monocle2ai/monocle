# Agentic Application Patterns and Framework Support

## Overview

Modern agentic AI applications follow various architectural patterns. This guide shows which patterns each framework supports and how Monocle instruments them.

## Agentic Patterns

- **Single Agent** - One agent handles all tasks
- **Multi-Agent Sequential** - Agents work in sequence, passing results
- **Multi-Agent Parallel** - Agents work simultaneously  
- **Orchestrator/Supervisor** - Coordinator agent manages specialized sub-agents
- **Agent Delegation/Handoffs** - Agents transfer control to other agents (different frameworks use different terms: `handoffs`, `handoff_to_*`, `transfer_to_*`)
- **Session/Thread** - Multi-turn conversation context preservation

## Framework Support and Monocle Instrumentation

| Framework | Single | Sequential | Parallel | Orchestrator | Delegation/Handoffs | Session/Thread | Monocle Support |
|-----------|:------:|:----------:|:--------:|:------------:|:-------------------:|:--------------:|:---------------:|
| **Google ADK** | ✅ | ✅<br/>SequentialAgent | ✅<br/>ParallelAgent | ✅<br/>LoopAgent | ✅<br/>Built-in | ✅<br/>`session_id` | **Full** ✅ |
| **LangGraph** | ✅ | ✅ | ✅ | ✅<br/>Supervisor | ✅<br/>`transfer_to_*` | ✅<br/>`thread_id` | **Full** ✅ |
| **OpenAI Agents** | ✅ | ✅ | ⚠️ | ✅ | ✅<br/>`handoffs=[]` | ✅<br/>`thread` | **Full** ✅ |
| **Microsoft Agent Framework** | ✅ | ✅ | ⚠️ | ✅ | ✅<br/>`handoff_to_*` | ✅<br/>`thread` | **Full** ✅ |
| **LlamaIndex** | ✅ | ✅ | ⚠️ | ✅ | ⚠️ | ✅<br/>`session_id` | **Partial** ⚠️ |
| **CrewAI** | ✅ | ✅<br/>Sequential | ✅<br/>Hierarchical | ✅ | ✅ | ⚠️ | **Partial** ⚠️ |
| **Strands** | ✅ | ✅ | ⚠️ | ⚠️ | ⚠️ | ✅<br/>`session_id` | **Partial** ⚠️ |
| **LangChain** | ✅ | ✅ | ✅ | ⚠️ | ⚠️ | ⚠️ | **Basic** ⚠️ |
| **Haystack** | ✅ | ✅<br/>Pipeline | ✅ | ⚠️ | ⚠️ | ⚠️ | **Basic** ⚠️ |

**Legend:** ✅ Full support | ⚠️ Partial/Limited | ❌ Not supported