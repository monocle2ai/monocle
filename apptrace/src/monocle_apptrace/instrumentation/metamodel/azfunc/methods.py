from monocle_apptrace.instrumentation.common.wrapper import atask_wrapper, task_wrapper
from monocle_apptrace.instrumentation.metamodel.azfunc.entities.http import AZFUNC_HTTP_PROCESSOR

AZFUNC_HTTP_METHODS = [
    {
        "package": "monocle_apptrace.instrumentation.metamodel.azfunc.wrapper",
        "object": "AzureFunctionRouteWrapper",
        "method": "run_async",
        "span_name": "azure_function_route",
        "wrapper_method": atask_wrapper,
        "span_handler": "azure_func_handler",
        "output_processor": AZFUNC_HTTP_PROCESSOR
    },
    {
        "package": "monocle_apptrace.instrumentation.metamodel.azfunc.wrapper",
        "object": "AzureFunctionRouteWrapper",
        "method": "run_sync",
        "span_name": "azure_function_route",
        "wrapper_method": task_wrapper,
        "span_handler": "azure_func_handler",
        "output_processor": AZFUNC_HTTP_PROCESSOR
    }
]
