"""
Claude Code framework method instrumentation definitions.

Claude Code is a CLI binary (not a Python library), so there are no methods
to monkey-patch. The METHODS list is empty. Instead, the transcript_processor
module is used directly by the Claude Code Stop hook to emit spans.
"""

CLAUDE_CODE_METHODS = []
