"""CLI tool to generate test code from a Monocle trace.

Input is one of:
    - a local trace file, or
    - an Okahu trace (--trace-id + --workflow-name), or
    - an Okahu agentic session (--session-id + --workflow-name), or
    - any other Okahu fact/scope (--fact-name + --fact-id + --workflow-name).

Usage:
    python -m monocle_test_tools generate_test --trace-file trace.json
    python -m monocle_test_tools generate_test --trace-id <id> --workflow-name <name>
    python -m monocle_test_tools generate_test --session-id <id> --workflow-name <name>
    python -m monocle_test_tools generate_test --fact-name test_id --fact-id <id> --workflow-name <name>

Generated tests assert on agents, tools, outputs, tokens, and turn duration.
The Okahu cloud loader is pre-populated with trace id and workflow name.

--eval [TYPE:]NAME_OR_PATH=EXPECTED   Inject eval assertions (repeatable).
  Type auto-detected (path -> custom, else built-in) or forced via 'builtin:'/'custom:'.

Evals already recorded on the source fact are auto-discovered and added as
check_eval() baseline assertions; disable with --no-discover-evals.
"""

import argparse
import sys


def _parse_eval_spec(raw: str, default_fact: str, eval_source: str = "okahu") -> dict:
    """Parse ``[TYPE:]NAME_OR_PATH=EXPECTED`` into an eval spec.

    A ``builtin:``/``custom:`` prefix forces the type, else it is auto-detected
    using ``eval_source``'s rules. ``=EXPECTED`` is required; raises ``ValueError``
    on malformed input (or an unknown ``eval_source``).
    """
    from monocle_test_tools.test_generator import TestGenerator

    explicit_type = None
    for candidate in ("builtin", "custom"):
        if raw.startswith(candidate + ":"):
            explicit_type, raw = candidate, raw[len(candidate) + 1:]
            break

    name_part, sep, expected = raw.partition("=")
    name_part = name_part.strip()
    if not name_part:
        raise ValueError("--eval requires an eval name or template path")
    if not sep or not expected.strip():
        raise ValueError(
            f"--eval '{raw}' must specify an expected result, e.g. "
            f"'{name_part}=<expected>'"
        )

    eval_type = explicit_type or TestGenerator._detect_eval_type(name_part, eval_source)
    spec: dict = {
        "fact_name": default_fact or "traces",
        "eval_type": eval_type,
        "expected": expected.strip(),
    }
    if eval_type == "custom":
        spec["template_path"] = name_part
    else:
        spec["criteria"] = name_part

    return spec


def main():
    parser = argparse.ArgumentParser(
        description="Generate test code from a Monocle trace or agentic session",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument("trace_file", nargs="?", default=None, help="Path to trace JSON file")
    parser.add_argument("--trace-file", dest="_trace_file_flag", metavar="TRACE_FILE", default=None, help="Path to trace JSON file")
    parser.add_argument("--trace-id", default=None, help="Okahu trace ID to fetch spans from")
    parser.add_argument("--session-id", default=None, help="Okahu agentic session ID to fetch all session spans from")
    parser.add_argument("--fact-name", default=None, help="Okahu fact/scope name to fetch spans by (e.g. test_id, conversations); use with --fact-id")
    parser.add_argument("--fact-id", default=None, help="Value of the --fact-name fact/scope")
    parser.add_argument("--workflow-name", default=None, help="Okahu workflow name (required with --trace-id / --session-id / --fact-name)")
    parser.add_argument("--test-name", default="test_generated", help="Test function name (default: test_generated)")
    
    parser.add_argument(
        "--trace-source",
        choices=["file", "okahu"],
        default=None,
        help="Only generate loader code for this trace source (file|okahu). "
             "If omitted, code for all sources is generated.",
    )
    parser.add_argument(
        "--eval",
        dest="evals",
        metavar="NAME_OR_PATH[=EXPECTED]",
        action="append",
        default=[],
        help=(
            "Inject an eval assertion (repeatable). Format: [TYPE:]NAME_OR_PATH=EXPECTED, "
            "e.g. hallucination=no_hallucination or custom:./my_eval.json=pass. "
            "Type auto-detected (path -> custom, else built-in) or forced via builtin:/custom: prefix."
        ),
    )
    parser.add_argument(
        "--eval-fact",
        default=None,
        help="Fact_name for injected --eval assertions (default: traces). Also overrides "
             "the auto-matched fact level used for eval discovery.",
    )
    parser.add_argument(
        "--eval-source",
        default=None,
        help="Evaluator for the generated with_evaluation(...) calls; also drives how "
             "eval names/paths are classified as built-in vs custom. "
             "Required when --eval is used (e.g. okahu).",
    )
    parser.add_argument(
        "--no-discover-evals",
        dest="discover_evals",
        action="store_false",
        help="Disable auto-discovery of evals already recorded on the source fact.",
    )

    args = parser.parse_args()
    
    trace_file = args._trace_file_flag if args._trace_file_flag else args.trace_file
    has_fact = bool(args.fact_name or args.fact_id)

    if args.evals and not args.eval_source:
        parser.error("--eval-source is required when --eval is used (e.g. --eval-source okahu)")

    # Exactly one input mode: file, okahu trace, okahu session, or okahu fact/scope.
    modes = [bool(trace_file), bool(args.trace_id), bool(args.session_id), has_fact]
    if sum(modes) > 1:
        parser.error("provide only one of: a trace file, --trace-id, --session-id, or --fact-name/--fact-id")
    if not any(modes):
        parser.error("input is required: a trace file, or --trace-id/--session-id/--fact-name with --workflow-name")
    if has_fact and not (args.fact_name and args.fact_id):
        parser.error("--fact-name and --fact-id must be provided together")
    if (args.trace_id or args.session_id or has_fact) and not args.workflow_name:
        parser.error("--workflow-name is required with --trace-id / --session-id / --fact-name")

    # Parse injected evals (malformed specs / unknown eval-source surface as a clean error).
    try:
        injected_evals = [_parse_eval_spec(raw, args.eval_fact, args.eval_source) for raw in args.evals]
    except ValueError as exc:
        parser.error(str(exc))

    from monocle_test_tools.test_generator import TestGenerator
    
    try:
        if args.session_id:
            generator = TestGenerator.from_okahu_session(
                session_id=args.session_id, workflow_name=args.workflow_name,
                trace_source=args.trace_source,
                injected_evals=injected_evals, eval_source=args.eval_source,
                discover_evals=args.discover_evals, discovery_fact_name=args.eval_fact,
            )
        elif has_fact:
            generator = TestGenerator.from_okahu_scope(
                scope_name=args.fact_name, scope_id=args.fact_id,
                workflow_name=args.workflow_name, trace_source=args.trace_source,
                injected_evals=injected_evals, eval_source=args.eval_source,
                discover_evals=args.discover_evals, discovery_fact_name=args.eval_fact,
            )
        elif args.trace_id:
            generator = TestGenerator.from_okahu(
                trace_id=args.trace_id, workflow_name=args.workflow_name,
                trace_source=args.trace_source,
                injected_evals=injected_evals, eval_source=args.eval_source,
                discover_evals=args.discover_evals, discovery_fact_name=args.eval_fact,
            )
        else:
            generator = TestGenerator.from_json_file(
                trace_file, trace_source=args.trace_source,
                injected_evals=injected_evals, eval_source=args.eval_source,
                discover_evals=args.discover_evals, discovery_fact_name=args.eval_fact,
            )
        test_code = generator.generate_test_code(test_name=args.test_name)
        note = getattr(generator, "_discovery_note", None)
        if note:
            print(f"Warning: {note}", file=sys.stderr)
        print(test_code)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
