from monocle_apptrace.instrumentation.common.git_context import GitContext
from monocle_apptrace.instrumentation.metamodel.claude_cli.trace_events import _sessions_dir

_ctx = GitContext(_sessions_dir, "monocle_claude")

capture_turn_baseline = _ctx.capture_turn_baseline
compute_scopes = _ctx.compute_scopes
cleanup = _ctx.cleanup
