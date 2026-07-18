"""CLI tool to generate test code from trace files.

Usage:
    python -m monocle_test_tools.generate_test trace.json
    python -m monocle_test_tools.generate_test .monocle/test_traces/trace_abc123.json
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
    args = parser.parse_args()
    
    trace_file = args._trace_file_flag if args._trace_file_flag else args.trace_file
    if not trace_file:
        parser.error("trace file is required: provide it as a positional argument or via --trace-file")
    
    from monocle_test_tools.test_generator import TestGenerator
    
    try:
        generator = TestGenerator.from_json_file(trace_file, trace_source=args.trace_source)
        test_code = generator.generate_test_code(test_name=args.test_name)
        print(test_code)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
