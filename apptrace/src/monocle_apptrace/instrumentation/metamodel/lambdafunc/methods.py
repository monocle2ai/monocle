from monocle_apptrace.instrumentation.common.wrapper import atask_wrapper, task_wrapper
from monocle_apptrace.instrumentation.metamodel.lambdafunc.entities.http import LAMBDA_HTTP_PROCESSOR

LAMBDA_HTTP_METHODS = [
    {
        "package": "monocle_apptrace.instrumentation.metamodel.lambdafunc.wrapper",
        "object": "LambdaFunctionRouteWrapper",
        "method": "run_async",
        "span_name": "lambda_function_route",
        "wrapper_method": atask_wrapper,
        "span_handler": "lambda_func_handler",
        "output_processor": LAMBDA_HTTP_PROCESSOR
    },
    {
        "package": "monocle_apptrace.instrumentation.metamodel.lambdafunc.wrapper",
        "object": "LambdaFunctionRouteWrapper",
        "method": "run_sync",
        "span_name": "lambda_function_route",
        "wrapper_method": task_wrapper,
        "span_handler": "lambda_func_handler",
        "output_processor": LAMBDA_HTTP_PROCESSOR
    }
]
