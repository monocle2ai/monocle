import sys, os
import runpy
from monocle_apptrace import setup_monocle_telemetry


def main():
    if len(sys.argv) >= 2:
        cmd = sys.argv[1]

        if cmd in ("claude-hook", "claude_hook", "--claude-hook"):
            from monocle_apptrace.instrumentation.metamodel.claude_cli.event_handler import main as hook_main
            hook_main()
            sys.exit(0)

        if cmd in ("claude-setup", "claude_setup"):
            from monocle_apptrace.instrumentation.metamodel.claude_cli.installer import install
            sys.exit(install())

    # Original behavior: wrap user scripts
    if len(sys.argv) < 2 or not sys.argv[1].endswith(".py"):
        print("Usage:")
        print("  python -m monocle_apptrace <your-main-module-file>   wrap a script with Monocle telemetry")
        print("  python -m monocle_apptrace claude-setup              register Claude Code hooks")
        print("  python -m monocle_apptrace claude-hook               read a hook event from stdin (manual testing)")
        sys.exit(1)
    
    file_name = os.path.basename(sys.argv[1])
    workflow_name = file_name[:-3]
    setup_monocle_telemetry(workflow_name=workflow_name)
    sys.argv.pop(0)

    try:
        runpy.run_path(path_name=sys.argv[0], run_name="__main__")
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()
