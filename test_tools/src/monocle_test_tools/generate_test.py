"""CLI tool to generate test code from trace files.

Usage:
    python -m monocle_test_tools generate_test --trace-file trace.json
    python -m monocle_test_tools generate_test --trace-id <id> --workflow-name <name>

Generated tests assert on agents, tools, outputs, tokens, and turn duration.
The Okahu cloud loader is pre-populated with trace id and workflow name.

--eval [TYPE:]NAME_OR_PATH=EXPECTED   Inject eval assertions (repeatable).
  Type auto-detected (path -> custom, else built-in) or forced via 'builtin:'/'custom:'.
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
        description="Generate test code from a Monocle trace file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("trace_file", nargs="?", default=None, help="Path to trace JSON file")
    parser.add_argument("--trace-file", dest="_trace_file_flag", metavar="TRACE_FILE", default=None, help="Path to trace JSON file")
    parser.add_argument("--trace-id", default=None, help="Okahu trace ID to fetch spans from")
    parser.add_argument("--workflow-name", default=None, help="Okahu workflow name for --trace-id")
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
        default="traces",
        help="Default fact_name for injected --eval assertions (default: traces).",
    )
    parser.add_argument(
        "--eval-source",
        default="okahu",
        help="Evaluator for the generated with_evaluation(...) calls; also drives how "
             "eval names/paths are classified as built-in vs custom (default: okahu).",
    )

    args = parser.parse_args()
    
    trace_file = args._trace_file_flag if args._trace_file_flag else args.trace_file
    has_file_input = bool(trace_file)
    has_okahu_input = bool(args.trace_id or args.workflow_name)

    if has_file_input and has_okahu_input:
        parser.error("provide either a trace file OR (--trace-id and --workflow-name), not both")

    if has_okahu_input and (not args.trace_id or not args.workflow_name):
        parser.error("both --trace-id and --workflow-name are required for Okahu source")

    if not has_file_input and not (args.trace_id and args.workflow_name):
        parser.error(
            "input is required: provide a trace file (positional or --trace-file) "
            "or provide both --trace-id and --workflow-name"
        )

    # Parse injected evals (malformed specs / unknown eval-source surface as a clean error).
    try:
        injected_evals = [_parse_eval_spec(raw, args.eval_fact, args.eval_source) for raw in args.evals]
    except ValueError as exc:
        parser.error(str(exc))

    from monocle_test_tools.test_generator import TestGenerator
    
    try:
        if has_file_input:
            generator = TestGenerator.from_json_file(
                trace_file,
                trace_source=args.trace_source,
                injected_evals=injected_evals,
                eval_source=args.eval_source,
            )
        else:
            generator = TestGenerator.from_okahu(
                trace_id=args.trace_id,
                workflow_name=args.workflow_name,
                trace_source=args.trace_source,
                injected_evals=injected_evals,
                eval_source=args.eval_source,
            )
        test_code = generator.generate_test_code(test_name=args.test_name)
        print(test_code)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
