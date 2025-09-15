from functools import wraps
import inspect

def monocle_trace_lambda_function_route(func):
    if inspect.iscoroutinefunction(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await LambdaFunctionRouteWrapper.run_async(func, *args, **kwargs)
        return wrapper
    else:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return LambdaFunctionRouteWrapper.run_sync(func, *args, **kwargs)
        return wrapper

class LambdaFunctionRouteWrapper:
    @staticmethod
    async def run_async(func, *args, **kwargs):
        return await func(*args, **kwargs)

    @staticmethod
    def run_sync(func, *args, **kwargs):
        return func(*args, **kwargs)