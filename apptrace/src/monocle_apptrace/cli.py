"""Monocle CLI — install hooks for AI coding agents and dispatch hook events.

Subcommands:
  claude-setup    register Monocle hooks for Claude Code
  codex-setup     register Monocle hooks for Codex CLI
  copilot-setup   register Monocle hooks for GitHub Copilot (CLI + VS Code Chat)
  validate        validate a trace file against the metamodel
  reset           dev helper — clear local state (REMOVE BEFORE PR)

Hook dispatch (`claude-hook`, `codex-hook`, `copilot-hook`) is invoked as a
subprocess by each agent when a hook fires — see `main()` for that path.
"""
import argparse
import getpass
import importlib.metadata
import json
import os
import stat
import sys
from pathlib import Path

from monocle_apptrace.auth import ui

# Global install → write env config to ~/.monocle/.env (shared across all
# sessions on this machine). Project install → write to ./.env.monocle in
# the project root (scoped to that directory only).
GLOBAL_ENV_FILE = Path.home() / ".monocle" / ".env"
PROJECT_ENV_FILENAME = ".env.monocle"

_METAMODEL_DIR = Path(__file__).parent / "instrumentation" / "metamodel"
_HOOK_MARKERS = ("monocle-apptrace", "monocle_apptrace.instrumentation.metamodel")
_DOCS_URL = "https://docs.okahu.ai/"
_VSCODE_EXTENSION_URL = "https://marketplace.visualstudio.com/items?itemName=OkahuAI.okahu-ai-observability"


# =============================================================================
# Setup commands  (claude-setup, codex-setup)
# =============================================================================


def cmd_claude_setup(args):
    return _run_setup(
        args,
        agent="claude",
        label="Claude Code",
        hooks_subpath=".claude/settings.json",
        hooks_template=_METAMODEL_DIR / "claude_cli" / "hooks.json",
    )


def cmd_codex_setup(args):
    return _run_setup(
        args,
        agent="codex",
        label="Codex CLI",
        hooks_subpath=".codex/hooks.json",
        hooks_template=_METAMODEL_DIR / "codex_cli" / "hooks.json",
        post_install=_enable_codex_hooks_flag,
    )


def cmd_copilot_setup(args):
    return _run_setup(
        args,
        agent="copilot",
        label="GitHub Copilot",
        # Project → <repo>/.github/hooks/; both Copilot CLI and VS Code Chat discover
        # .github/hooks/*.json in the workspace by default. Global → ~/.copilot/hooks/.
        hooks_subpath=".github/hooks/monocle.json",
        global_hooks_subpath=".copilot/hooks/monocle.json",
        hooks_template=_METAMODEL_DIR / "github_copilot" / "hooks.json",
        post_install=_configure_copilot_otel,
    )


def _configure_copilot_otel(scope_root):
    """Enable Copilot's native OTel file export (VS Code Chat + Copilot CLI) so replay
    can read token counts. Fixed path (~/.monocle/.copilot_otel/copilot.jsonl) because
    VS Code only reliably file-exports to a stable outfile."""
    otel_dir = Path.home() / ".monocle" / ".copilot_otel"
    otel_dir.mkdir(parents=True, exist_ok=True)
    outfile = str(otel_dir / "copilot.jsonl")

    # Record the path so replay finds it regardless of cwd/surface (VS Code hooks
    # don't inherit the shell env, unlike the CLI).
    env_file = GLOBAL_ENV_FILE if scope_root == Path.home() else scope_root / PROJECT_ENV_FILENAME
    _save_env(env_file, MONOCLE_COPILOT_OTEL_FILE=outfile)

    vscode_settings = _vscode_settings_path()
    if vscode_settings is not None:
        try:
            settings = json.loads(vscode_settings.read_text()) if vscode_settings.exists() else {}
        except Exception:
            settings = {}
        settings["github.copilot.chat.otel.enabled"] = True
        settings["github.copilot.chat.otel.exporterType"] = "file"
        settings["github.copilot.chat.otel.outfile"] = outfile
        vscode_settings.parent.mkdir(parents=True, exist_ok=True)
        vscode_settings.write_text(json.dumps(settings, indent=2) + "\n")
        ui.hint("VS Code OTel export enabled: " + str(vscode_settings))

    rc_path = _shell_rc_path()
    if rc_path is not None:
        _ensure_copilot_otel_rc_block(rc_path, outfile)
        ui.hint("Shell env updated for Copilot CLI: " + str(rc_path))


def _vscode_settings_path():
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Code" / "User" / "settings.json"
    if sys.platform.startswith("linux"):
        return Path.home() / ".config" / "Code" / "User" / "settings.json"
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        return Path(appdata) / "Code" / "User" / "settings.json" if appdata else None
    return None


def _shell_rc_path():
    shell = os.environ.get("SHELL", "")
    if shell.endswith("zsh"):
        return Path.home() / ".zshrc"
    if shell.endswith("bash"):
        return Path.home() / (".bash_profile" if sys.platform == "darwin" else ".bashrc")
    return None


_COPILOT_OTEL_MARKER = "# >>> monocle copilot otel >>>"
_COPILOT_OTEL_END = "# <<< monocle copilot otel <<<"


def _ensure_copilot_otel_rc_block(rc_path, outfile):
    block = "\n".join([
        _COPILOT_OTEL_MARKER,
        "export COPILOT_OTEL_ENABLED=true",
        "export COPILOT_OTEL_EXPORTER_TYPE=file",  # CLI ignores the file path without this
        "export COPILOT_OTEL_FILE_EXPORTER_PATH={}".format(outfile),
        _COPILOT_OTEL_END,
        "",
    ])
    existing = rc_path.read_text() if rc_path.exists() else ""
    if _COPILOT_OTEL_MARKER in existing:
        before = existing.split(_COPILOT_OTEL_MARKER)[0]
        after = existing.split(_COPILOT_OTEL_END, 1)
        tail = after[1] if len(after) > 1 else ""
        rc_path.write_text(before.rstrip() + "\n\n" + block + tail.lstrip())
    else:
        sep = "\n" if existing and not existing.endswith("\n") else ""
        rc_path.write_text(existing + sep + "\n" + block)


def _run_setup(args, *, agent, label, hooks_subpath, hooks_template, post_install=None, global_hooks_subpath=None):
    ui.header("Monocle setup", label)

    # 1. Where to install: ~ (global) or cwd (project). global_hooks_subpath
    #    overrides the global hook dir when it differs from the project one (Copilot).
    scope = args.scope or _prompt_install_scope(label)
    if scope == "global":
        root = Path.home()
        env_file = GLOBAL_ENV_FILE
        settings_file = Path.home() / (global_hooks_subpath or hooks_subpath)
    else:
        root = Path.cwd()
        env_file = root / PROJECT_ENV_FILENAME
        settings_file = root / hooks_subpath

    # 2. Get the API key (or None if the user chose local-only / skip).
    api_key, aborted = _get_api_key(args)
    if aborted:
        return 1

    # 3. Save env and install hooks.
    if api_key:
        _save_env(env_file, OKAHU_API_KEY=api_key, MONOCLE_EXPORTER="okahu")
    else:
        _save_env(env_file, MONOCLE_EXPORTER="file")
    _install_hooks(agent, settings_file, hooks_template)
    if post_install:
        post_install(root)

    # 4. Done.
    ui.check("Hooks installed for " + label)
    ui.blank()
    ui.hint("Config: " + str(env_file))
    ui.hint("Hooks:  " + str(settings_file))
    ui.blank()
    ui.step(ui.brand("Monocle is ready.") + " Run " + ui.bold(agent) + " to start tracing.")
    ui.blank()
    return 0


# =============================================================================
# API key resolution
# =============================================================================


def _get_api_key(args):
    """Decide how to get an OKAHU_API_KEY.

    Returns (api_key, aborted):
      api_key is the key string, or None for "no key — use file exporter."
      aborted is True only if the user tried to sign in and it failed.
    """
    from monocle_apptrace.auth.okahu_api import OkahuApiError, is_max_api_keys_reached
    from monocle_apptrace.auth.portal_auth import PortalAuthError, resolve_okahu_api_key

    if args.api_key:
        return args.api_key, False
    if args.local_only:
        return None, False

    existing = _read_existing_key()

    # Pick the sign-in mechanism: explicit flag wins; otherwise reuse the
    # existing key (with confirmation if interactive); otherwise prompt.
    if args.portal:
        method = "loopback"
    elif args.device:
        method = "device"
    elif existing and not sys.stdin.isatty():
        return existing, False  # CI / scripts: reuse silently
    elif existing and _confirm_reuse_existing(existing):
        return existing, False
    else:
        method = _prompt_auth_method()

    if method in ("smart", "loopback", "device"):
        try:
            result = resolve_okahu_api_key(method=method)
        except PortalAuthError as e:
            ui.fail("Sign-in failed: " + str(e))
            return None, True
        except OkahuApiError as e:
            if is_max_api_keys_reached(e):
                _explain_max_keys_reached()
            else:
                ui.fail("Okahu API error: " + str(e))
            return None, True
        return result["api_key"], False

    if method == "paste":
        key = getpass.getpass("  Okahu API key: ").strip()
        return (key or None), False

    return None, False  # "skip" or anything else falls through here.


def _read_existing_key():
    # Lazy import: get_monocle_env_value transitively pulls in OpenTelemetry,
    # which we don't want to pay for on `monocle-apptrace --version`.
    from monocle_apptrace.instrumentation.common.utils import get_monocle_env_value
    return get_monocle_env_value("OKAHU_API_KEY")


def _explain_max_keys_reached():
    """User-friendly message for the '8 API keys per tenant' cap."""
    ui.fail("Couldn't generate a new API key — your Okahu tenant has reached the 8-key limit.")
    ui.blank()
    ui.hint("Two ways to recover:")
    ui.hint("  • Reuse one of your existing keys — re-run setup and pick \"Paste an API key\".")
    ui.hint("  • Delete unused keys at https://portal.okahu.co (Settings → API Keys), then re-run.")


def _confirm_reuse_existing(existing_key):
    masked = existing_key[:6] + "…" + existing_key[-4:] if len(existing_key) > 10 else "…"
    if ui.confirm("Continue with existing API key " + ui.dim(masked) + "?", default_yes=True):
        ui.check("Using existing Okahu API key")
        return True
    return False


# =============================================================================
# Interactive prompts
# =============================================================================


def _prompt_install_scope(label):
    if not sys.stdin.isatty():
        return "global"
    return ui.select("Where should " + label + " hooks be installed?", [
        {"key": "global",  "label": "Global",
         "hint": str(Path.home()) + "  —  all " + label + " sessions on this machine"},
        {"key": "project", "label": "This project",
         "hint": str(Path.cwd()) + "  —  only sessions started here"},
    ])


def _prompt_auth_method():
    if not sys.stdin.isatty():
        return "skip"
    return ui.select(
        "How would you like to authenticate with Okahu?",
        [
            {"key": "smart", "label": "Sign in",
             "hint": "opens your browser, or a code if needed"},
            {"key": "paste", "label": "Paste an API key",
             "hint": "if you already have one from the Okahu portal"},
            {"key": "skip",  "label": "Skip",
             "hint": "store traces locally and inspect with the VS Code Okahu extension"},
        ],
        links=[
            ("learn more", _DOCS_URL),
            ("extension", _VSCODE_EXTENSION_URL),
        ],
    )


# =============================================================================
# File writers
# =============================================================================


def _save_env(env_file, **kwargs):
    """Update `env_file`. Each kwarg becomes a `KEY=value` line; existing
    lines for the same key are replaced in place. File mode is 0600."""
    env_file.parent.mkdir(parents=True, exist_ok=True)
    lines = env_file.read_text().splitlines() if env_file.exists() else [
        "# Monocle config — edit directly to change settings"
    ]
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
    env_file.write_text("\n".join(lines) + "\n")
    os.chmod(env_file, stat.S_IRUSR | stat.S_IWUSR)


def _install_hooks(agent, settings_file, hooks_template):
    """Merge Monocle hook entries into the agent's settings file (idempotent —
    re-running just updates the command if it's already there)."""
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


def _enable_codex_hooks_flag(scope_root):
    """Ensure `[features] hooks = true` in Codex's config.toml so hooks fire.

    `hooks` is the canonical key; `codex_hooks` is a deprecated alias that makes
    Codex print a warning, so migrate it in place when present."""
    config = scope_root / ".codex" / "config.toml"
    config.parent.mkdir(parents=True, exist_ok=True)
    text = config.read_text() if config.exists() else ""
    if "codex_hooks = true" in text:
        config.write_text(text.replace("codex_hooks = true", "hooks = true"))
        return
    if "hooks = true" in text:
        return
    sep = "\n" if text and not text.endswith("\n") else ""
    config.write_text(text + sep + "[features]\nhooks = true\n")


# =============================================================================
# Hook dispatch  (called by the agent's hook subprocess at runtime)
# =============================================================================


def hook_dispatch(agent):
    if agent == "claude":
        from monocle_apptrace.instrumentation.metamodel.claude_cli.event_handler import main as run
    elif agent == "codex":
        from monocle_apptrace.instrumentation.metamodel.codex_cli.event_handler import main as run
    elif agent == "copilot":
        from monocle_apptrace.instrumentation.metamodel.github_copilot.event_handler import main as run
    else:
        print("Unknown agent: {}".format(agent), file=sys.stderr)
        return 1
    run()
    return 0


# =============================================================================
# validate  (lint a trace file against the metamodel)
# =============================================================================


def cmd_validate(args):
    try:
        from monocle_apptrace.linter import MonocleValidator, ValidationReporter
        validator = MonocleValidator()
        results = validator.validate_trace_file(Path(args.trace_file))
        reporter = ValidationReporter()
        print(reporter.format_results(results, args.fail_on_warning))
        return reporter.get_exit_code(results, args.fail_on_warning)
    except FileNotFoundError as e:
        print("ERROR: {}".format(e), file=sys.stderr)
        return 1
    except Exception as e:
        print("ERROR: {}".format(e), file=sys.stderr)
        return 1


# =============================================================================
# DEV HELPER
# `monocle-apptrace reset` wipes ~/.monocle/.env, ~/.monocle/auth.json, and
# the Monocle hook entries in each agent's settings file. Lets us start
# fresh between test runs during development.
# =============================================================================


def cmd_token_summary(args):
    from monocle_apptrace.token_summary import format_table, summarize
    from pathlib import Path

    if args.trace_dir_direct:
        monocle_dir = Path(args.trace_dir_direct)
    elif args.trace_dir:
        monocle_dir = Path(args.trace_dir) / ".monocle"
    else:
        monocle_dir = None  # uses default ~/.monocle

    rows = summarize(time_window=args.time_window, monocle_dir=monocle_dir)
    print(format_table(rows))
    return 0

def cmd_session_token_summary(args):
    from monocle_apptrace.session_token_summary import format_table, summarize
    from pathlib import Path

    if getattr(args, "trace_dir_direct", None):
        monocle_dir = Path(args.trace_dir_direct)
    elif getattr(args, "trace_dir", None):
        monocle_dir = Path(args.trace_dir) / ".monocle"
    else:
        monocle_dir = None

    rows = summarize(time_window=args.time_window, monocle_dir=monocle_dir)
    print(format_table(rows))
    return 0


def cmd_reset(args):
    ui.header("Monocle reset")

    cleared = []
    for path in (
        GLOBAL_ENV_FILE,
        Path.home() / ".monocle" / "auth.json",
        Path.cwd() / PROJECT_ENV_FILENAME,  # project-level env, if reset is run inside a project
    ):
        if path.exists():
            path.unlink()
            cleared.append(path)

    hook_paths = (
        Path.home() / ".claude" / "settings.json",
        Path.cwd()  / ".claude" / "settings.json",
        Path.home() / ".codex"  / "hooks.json",
        Path.cwd()  / ".codex"  / "hooks.json",
    )
    for path in hook_paths:
        if path.exists() and _strip_monocle_hooks(path):
            cleared.append(path)

    if cleared:
        for p in cleared:
            ui.check("Cleaned " + str(p))
    else:
        ui.step("Nothing to clean.")
    ui.blank()
    ui.step(ui.brand("Reset done.") + " Run " + ui.bold("monocle-apptrace claude-setup") + " for a fresh setup.")
    ui.blank()
    return 0


def _strip_monocle_hooks(settings_file):
    """Remove Monocle hook entries from a settings file. Returns True if changed."""
    try:
        settings = json.loads(settings_file.read_text())
    except Exception:
        return False
    hooks = settings.get("hooks", {})
    if not hooks:
        return False

    changed = False
    for event in list(hooks.keys()):
        new_groups = []
        for group in hooks.get(event, []):
            inner = group.get("hooks", [])
            kept = [h for h in inner if not any(m in h.get("command", "") for m in _HOOK_MARKERS)]
            if kept != inner:
                changed = True
            if kept:
                group["hooks"] = kept
                new_groups.append(group)
            elif inner:
                continue  # all inner hooks were monocle — drop the group
            else:
                # Copilot-style: command at the group level.
                if any(m in group.get("command", "") for m in _HOOK_MARKERS):
                    changed = True
                else:
                    new_groups.append(group)
        if new_groups:
            hooks[event] = new_groups
        else:
            del hooks[event]
            changed = True
    if not hooks:
        settings.pop("hooks", None)
    if changed:
        settings_file.write_text(json.dumps(settings, indent=2))
    return changed


# =============================================================================
# Entry point
# =============================================================================


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)

    # Hook subprocess paths — invoked by the agent at runtime, no argparse needed.
    if argv and argv[0] == "claude-hook":
        return hook_dispatch("claude")
    if argv and argv[0] == "codex-hook":
        return hook_dispatch("codex")
    if argv and argv[0] == "copilot-hook":
        return hook_dispatch("copilot")

    if not argv:
        print("monocle-apptrace {}".format(_package_version()))
        return 0

    args = _build_parser().parse_args(argv)

    if args.command == "claude-setup":
        return cmd_claude_setup(args)
    if args.command == "codex-setup":
        return cmd_codex_setup(args)
    if args.command == "copilot-setup":
        return cmd_copilot_setup(args)
    if args.command == "validate":
        return cmd_validate(args)
    if args.command == "reset":
        return cmd_reset(args)
    if args.command == "token-summary":
        return cmd_token_summary(args)
    if args.command == "session-token-summary":
        return cmd_session_token_summary(args)
    return 1


def _package_version():
    try:
        return importlib.metadata.version("monocle_apptrace")
    except importlib.metadata.PackageNotFoundError:
        return "dev"


def _build_parser():
    parser = argparse.ArgumentParser(prog="monocle-apptrace")
    parser.add_argument("--version", action="version",
                        version="monocle-apptrace {}".format(_package_version()))
    sub = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    # claude-setup and codex-setup share the same flags.
    for name, help_text in (("claude-setup", "Register Monocle hooks for Claude Code"),
                            ("codex-setup",  "Register Monocle hooks for Codex CLI"),
                            ("copilot-setup", "Register Monocle hooks for GitHub Copilot (CLI + VS Code)")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--api-key", help="Use this API key without prompting")
        p.add_argument("--portal", action="store_true",
                       help="Sign in via browser (skip the prompt)")
        p.add_argument("--device", action="store_true",
                       help="Sign in with a code (no callback URL; works over SSH)")
        p.add_argument("--local-only", action="store_true",
                       help="Write traces to file only, no Okahu key needed")
        scope_group = p.add_mutually_exclusive_group()
        scope_group.add_argument("--global", dest="scope", action="store_const", const="global",
                                 help="Install hooks at ~/.<agent>/ (all sessions on this machine)")
        scope_group.add_argument("--project", dest="scope", action="store_const", const="project",
                                 help="Install hooks at ./.<agent>/ (only this directory)")

    sub.add_parser("reset", help="Clear Monocle state (dev helper; will be removed before PR)")

    v = sub.add_parser("validate", help="Validate Monocle traces against metamodel conformance")
    v.add_argument("trace_file", help="Path to trace JSON file")
    v.add_argument("--level", choices=["basic", "selective", "strict"],
                   default="selective", help="Validation level (default: selective)")
    v.add_argument("--fail-on-warning", action="store_true",
                   help="Treat warnings as errors (exit code 1)")

    ts = sub.add_parser(
        "token-summary",
        help="Show daily token usage from local .monocle/ trace files",
    )
    ts.add_argument(
        "time_window",
        nargs="?",
        default="all",
        metavar="TIME_WINDOW",
        help="today | 'this week' | '7 days' | '15 days' | all  (default: all)",
    )
    ts.add_argument(
        "--dir",
        dest="trace_dir",
        default=None,
        metavar="DIR",
        help="Project directory containing .monocle/ folder (default: current dir)",
    )
    ts.add_argument(
        "--trace-dir",
        dest="trace_dir_direct",
        default=None,
        metavar="TRACE_DIR",
        help="Direct path to trace directory (overrides --dir)",
    )

    sts = sub.add_parser(
        "session-token-summary",
        help="Show per-session token usage from local .monocle/ trace files",
    )
    sts.add_argument(
        "time_window",
        nargs="?",
        default="all",
        metavar="TIME_WINDOW",
        help="today | 'this week' | '7 days' | '15 days' | all  (default: all)",
    )
    sts.add_argument(
        "--dir",
        dest="trace_dir",
        default=None,
        metavar="DIR",
        help="Project directory containing .monocle/ folder (default: current dir)",
    )
    sts.add_argument(
        "--trace-dir",
        dest="trace_dir_direct",
        default=None,
        metavar="TRACE_DIR",
        help="Direct path to trace directory (overrides --dir)",
    )

    return parser


if __name__ == "__main__":
    sys.exit(main())
