import sys, os
import runpy
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry

def main():
    if len(sys.argv) < 2 or not sys.argv[1].endswith(".py"):
        print("Usage: python -m monocle_apptrace <your-main-module-file> <args>")
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