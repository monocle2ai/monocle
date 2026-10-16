"""Entry point for python -m monocle_test_tools [command] [options]"""

import sys
import importlib

SUBCOMMANDS = {
    "generate_test": "monocle_test_tools.generate_test",
}

HELP_TEXT = """Usage: python -m monocle_test_tools <command> [options]

Available commands:
  generate_test  —  Generate test code from a trace file
"""


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(HELP_TEXT, file=sys.stderr)
        sys.exit(1 if len(sys.argv) < 2 else 0)

    command = sys.argv.pop(1)
    module_path = SUBCOMMANDS.get(command)
    if module_path is None:
        print(
            f"Unknown command: {command}\n"
            f"Available: {', '.join(SUBCOMMANDS)}",
            file=sys.stderr,
        )
        sys.exit(1)

    module = importlib.import_module(module_path)
    sys.exit(module.main())


if __name__ == "__main__":
    main()
