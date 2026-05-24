import importlib
import sys, os
import runpy
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from monocle_apptrace.instrumentation.common.constants import WORKFLOW_NAME_ENV
USAGE = "Usage: python -m monocle_apptrace [--main <startup-function>] <your-main-module-file> <args>"

def main():
    if len(sys.argv) < 2:
        print(USAGE)
        sys.exit(1)

    # check if the there's an argument --main <function>, if so then extract the function name and call that instead of main
    main_func_name = "main"
    if sys.argv[1] == "--main":
        sys.argv.pop(1)  # Remove --main
        if len(sys.argv) < 3: # either function or the module name is missing
            print(USAGE)
            sys.exit(1)
        main_func_name = sys.argv[1]
        sys.argv.pop(1)  # Remove function name

    if len(sys.argv) < 2 or not sys.argv[1].endswith(".py"):
        print(USAGE)
        sys.exit(1)

    script_path = os.path.abspath(sys.argv[1])
    script_dir = os.path.dirname(script_path)
    module_name = os.path.basename(script_path)[:-3]
    workflow_name = os.getenv(WORKFLOW_NAME_ENV, module_name)
    sys.argv.pop(0)

    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    user_module = importlib.import_module(module_name)
    setup_monocle_telemetry(workflow_name=workflow_name)


    if hasattr(user_module, main_func_name) and callable(getattr(user_module, main_func_name)):
        getattr(user_module, main_func_name)()
    else:
        print(f"{module_name} has no callable {main_func_name}(). Please provide a valid main function using --main <function> or ensure your script has a main() function.")
        sys.exit(1)

if __name__ == "__main__":
    main()
