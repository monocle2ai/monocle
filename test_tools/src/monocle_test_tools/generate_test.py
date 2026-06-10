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
    
    parser.add_argument("trace_file", help="Path to trace JSON file")
    parser.add_argument("--test-name", default="test_generated", help="Test function name (default: test_generated)")
    
    args = parser.parse_args()
    
    from monocle_test_tools.test_generator import TestGenerator
    
    try:
        generator = TestGenerator.from_json_file(args.trace_file)
        test_code = generator.generate_test_code(test_name=args.test_name)
        print(test_code)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
