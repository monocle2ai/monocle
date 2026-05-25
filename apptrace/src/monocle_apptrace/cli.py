"""Monocle CLI — install hooks for AI coding agents and dispatch hook events.
"""
import argparse
import getpass
import importlib.metadata
import json
import os
import stat
import sys
from pathlib import Path

MONOCLE_ENV = Path.home() / ".monocle" / ".env"
_METAMODEL_DIR = Path(__file__).parent / "instrumentation" / "metamodel"
_HOOK_MARKERS = ("monocle-apptrace", "monocle_apptrace.instrumentation.metamodel")


def _enable_codex_hooks_flag():
    config = Path.home() / ".codex" / "config.toml"
    config.parent.mkdir(parents=True, exist_ok=True)
    text = config.read_text() if config.exists() else ""
    if "codex_hooks = true" in text:
        return
    sep = "\n" if text and not text.endswith("\n") else ""
    config.write_text(text + sep + "[features]\ncodex_hooks = true\n")


def _package_version():
    try:
        return importlib.metadata.version("monocle_apptrace")
    except importlib.metadata.PackageNotFoundError:
        return "dev"


def _save_env(**kwargs):
    MONOCLE_ENV.parent.mkdir(parents=True, exist_ok=True)
    lines = MONOCLE_ENV.read_text().splitlines() if MONOCLE_ENV.exists() else ["# Monocle config — edit directly to change settings"]
    for key, value in kwargs.items():
        replaced = False
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("export "):
                stripped = stripped[7:].strip()
            if stripped.startswith("{}=".format(key)):
                lines[i] = "{}={}".format(key, value)
                replaced = True
                break
        if not replaced:
            lines.append("{}={}".format(key, value))
    MONOCLE_ENV.write_text("\n".join(lines) + "\n")
    os.chmod(MONOCLE_ENV, stat.S_IRUSR | stat.S_IWUSR)


def setup(agent, label, settings_file, hooks_template, extra_setup=None):
    from monocle_apptrace.instrumentation.common.utils import get_monocle_env_value

    api_key = get_monocle_env_value("OKAHU_API_KEY")
    if api_key:
        print("Using existing Okahu API key.")
    else:
        api_key = getpass.getpass("Okahu API key (leave blank for local-only): ").strip()

    if api_key:
        _save_env(OKAHU_API_KEY=api_key, MONOCLE_EXPORTER="okahu")
    else:
        _save_env(MONOCLE_EXPORTER="file")

    _install_hooks(agent, settings_file, hooks_template, extra_setup)

    print("Monocle hooks installed for {}.".format(label))
    print("Config: {}".format(MONOCLE_ENV))
    print("Hooks:  {}".format(settings_file))
    return 0


def _install_hooks(agent, settings_file, hooks_template, extra_setup=None):
    template = json.loads(hooks_template.read_text())["hooks"]
    new_command = "monocle-apptrace {}-hook".format(agent)

    settings_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        settings = json.loads(settings_file.read_text()) if settings_file.exists() else {}
    except Exception:
        settings = {}
    settings.setdefault("hooks", {})

    for event, groups in template.items():
        already_registered = False
        for group in settings["hooks"].get(event, []):
            for hook in group.get("hooks", []):
                if any(m in hook.get("command", "") for m in _HOOK_MARKERS):
                    hook["command"] = new_command
                    already_registered = True
        if not already_registered:
            settings["hooks"].setdefault(event, []).extend(groups)

    settings_file.write_text(json.dumps(settings, indent=2))
    if extra_setup:
        extra_setup()


def hook_dispatch(agent):
    if agent == "claude":
        from monocle_apptrace.instrumentation.metamodel.claude_cli.event_handler import main as run
    elif agent == "codex":
        from monocle_apptrace.instrumentation.metamodel.codex_cli.event_handler import main as run
    else:
        print("Unknown agent: {}".format(agent), file=sys.stderr)
        return 1
    run()
    return 0


def cmd_validate(trace_file, level="selective", fail_on_warning=False):
    """Validate a Monocle trace file"""
    try:
        from monocle_apptrace.linter import MonocleValidator, ValidationReporter

        validator = MonocleValidator()
        results = validator.validate_trace_file(Path(trace_file))

        # Format and print output
        reporter = ValidationReporter()
        output = reporter.format_results(results, fail_on_warning)
        print(output)

        # Return exit code
        return reporter.get_exit_code(results, fail_on_warning)

    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)

    if argv and argv[0] == "claude-hook":
        return hook_dispatch("claude")
    if argv and argv[0] == "codex-hook":
        return hook_dispatch("codex")

    if not argv:
        print("monocle-apptrace {}".format(_package_version()))
        return 0

    parser = argparse.ArgumentParser(prog="monocle-apptrace")
    parser.add_argument("--version", action="version",
                        version="monocle-apptrace {}".format(_package_version()))
    sub = parser.add_subparsers(dest="command", required=True, metavar="<command>")
    sub.add_parser("claude-setup", help="Register Monocle hooks for Claude Code")
    sub.add_parser("codex-setup",  help="Register Monocle hooks for Codex CLI")

    # Add validate subcommand
    validate_parser = sub.add_parser("validate", help="Validate Monocle traces against metamodel conformance")
    validate_parser.add_argument("trace_file", help="Path to trace JSON file")
    validate_parser.add_argument("--level", choices=["basic", "selective", "strict"],
                                default="selective", help="Validation level (default: selective)")
    validate_parser.add_argument("--fail-on-warning", action="store_true",
                                help="Treat warnings as errors (exit code 1)")

    args = parser.parse_args(argv)
    if args.command == "claude-setup":
        return setup("claude", "Claude Code",
                     Path.home() / ".claude" / "settings.json",
                     _METAMODEL_DIR / "claude_cli" / "hooks.json")
    if args.command == "codex-setup":
        return setup("codex", "Codex CLI",
                     Path.home() / ".codex" / "hooks.json",
                     _METAMODEL_DIR / "codex_cli" / "hooks.json",
                     _enable_codex_hooks_flag)
    if args.command == "validate":
        return cmd_validate(args.trace_file, args.level, args.fail_on_warning)
    return 1


if __name__ == "__main__":
    sys.exit(main())
