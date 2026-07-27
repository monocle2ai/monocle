"""CLI tool to generate test code from trace files.

Usage:
    python -m monocle_test_tools.generate_test trace.json
    python -m monocle_test_tools.generate_test .monocle/test_traces/trace_abc123.json
    python -m monocle_test_tools.generate_test trace.json --include-io --include-attributes
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Generate test code from a Monocle trace file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument("trace_file", nargs="?", default=None, help="Path to trace JSON file")
    parser.add_argument("--trace-file", dest="_trace_file_flag", metavar="TRACE_FILE", default=None, help="Path to trace JSON file")
    parser.add_argument("--test-name", default="test_generated", help="Test function name (default: test_generated)")

    parser.add_argument(
        "--trace-source",
        choices=["file", "okahu"],
        default=None,
        help="Only generate loader code for this trace source (file|okahu). "
             "If omitted, code for all sources is generated.",
    )
    parser.add_argument(
        "--include-attributes",
        action="store_true",
        default=False,
        help="Emit has_attribute() assertions for notable span attributes "
             "(entity.1.type, workflow.name, etc). Off by default — these "
             "checks can be brittle as framework internals change.",
    )
    parser.add_argument(
        "--include-io",
        action="store_true",
        default=False,
        help="Emit has_input() / has_output() / contains_output() assertions "
             "for agent and tool I/O. Off by default — LLM outputs are "
             "non-deterministic and will vary across runs.",
    )
    args = parser.parse_args()

    trace_file = args._trace_file_flag if args._trace_file_flag else args.trace_file
    if not trace_file:
        parser.error("trace file is required: provide it as a positional argument or via --trace-file")

    from monocle_test_tools.test_generator import TestGenerator

    try:
        generator = TestGenerator.from_json_file(trace_file, trace_source=args.trace_source)
        test_code = generator.generate_test_code(
            test_name=args.test_name,
            include_attributes=args.include_attributes,
            include_io=args.include_io,
        )
        print(test_code)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
