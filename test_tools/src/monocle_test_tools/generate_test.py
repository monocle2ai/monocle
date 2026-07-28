"""CLI tool to generate test code from a Monocle trace.

Input is one of:
    - a local trace file, or
    - an Okahu trace (--trace-id + --workflow-name), or
    - an Okahu agentic session (--session-id + --workflow-name), or
    - any other Okahu fact/scope (--fact-name + --fact-id + --workflow-name).

Usage:
    python -m monocle_test_tools.generate_test trace.json
    python -m monocle_test_tools.generate_test --trace-id <id> --workflow-name <name>
    python -m monocle_test_tools.generate_test --session-id <id> --workflow-name <name>
    python -m monocle_test_tools.generate_test --fact-name test_id --fact-id <id> --workflow-name <name>
"""

import argparse
import sys


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
    args = parser.parse_args()
    
    trace_file = args._trace_file_flag if args._trace_file_flag else args.trace_file
    has_fact = bool(args.fact_name or args.fact_id)

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

    from monocle_test_tools.test_generator import TestGenerator
    
    try:
        if args.session_id:
            generator = TestGenerator.from_okahu_session(
                session_id=args.session_id, workflow_name=args.workflow_name,
                trace_source=args.trace_source,
            )
        elif has_fact:
            generator = TestGenerator.from_okahu_scope(
                scope_name=args.fact_name, scope_id=args.fact_id,
                workflow_name=args.workflow_name, trace_source=args.trace_source,
            )
        elif args.trace_id:
            generator = TestGenerator.from_okahu(
                trace_id=args.trace_id, workflow_name=args.workflow_name,
                trace_source=args.trace_source,
            )
        else:
            generator = TestGenerator.from_json_file(trace_file, trace_source=args.trace_source)
        test_code = generator.generate_test_code(test_name=args.test_name)
        print(test_code)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
