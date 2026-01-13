from monocle_apptrace.instrumentation.common.utils import get_exception_status_code

def get_route(args):
    return getattr(args[0], "__name__", "unknown")

def get_inputs(args):
    return str(args[3]) if len(args) > 3 else None

def extract_status(arguments):
    if arguments["exception"] is not None:
        return get_exception_status_code(arguments)
    return "success"
 
def get_outputs(result):
    return str(result)
