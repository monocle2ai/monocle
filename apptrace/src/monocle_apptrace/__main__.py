import importlib
import sys, os
import runpy
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from monocle_apptrace.instrumentation.common.constants import WORKFLOW_NAME_ENV

def main():
    if len(sys.argv) < 2 or not sys.argv[1].endswith(".py"):
        print("Usage: python -m monocle_apptrace <your-main-module-file> <args>")
        sys.exit(1)

    script_path = os.path.abspath(sys.argv[1])
    script_dir = os.path.dirname(script_path)
    mod_name = os.path.basename(script_path)[:-3]
    workflow_name = os.getenv(WORKFLOW_NAME_ENV, mod_name)

    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    user_module = importlib.import_module(mod_name)
    setup_monocle_telemetry(workflow_name=workflow_name)

    sys.argv.pop(0)
    if hasattr(user_module, "main") and callable(user_module.main):
        user_module.main()
    else:
        print(f"{mod_name} has no callable main(); cannot run in this mode. Either add a main() or call "
              f"setup_monocle_telemetry() from inside the script.")
        sys.exit(1)

if __name__ == "__main__":
    main()
