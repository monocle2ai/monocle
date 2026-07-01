from monocle_apptrace.instrumentation.common.agent_edit_context import GitContext
from monocle_apptrace.instrumentation.metamodel.codex_cli.trace_events import _sessions_dir

# Codex only needs the live git snapshot (repo/branch/uncommitted/ahead-behind).
# Per-turn line deltas come from the transcript patches (replay._turn_line_stats),
# so there is no turn baseline to capture and nothing to clean up here.
_ctx = GitContext(_sessions_dir, "monocle_codex")

compute_scopes = _ctx.compute_scopes
