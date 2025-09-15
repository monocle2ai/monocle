from functools import wraps
import inspect

def monocle_trace_azure_function_route(func):
    if inspect.iscoroutinefunction(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await AzureFunctionRouteWrapper.run_async(func, *args, **kwargs)
        return wrapper
    else:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return AzureFunctionRouteWrapper.run_sync(func ,*args, **kwargs)
        return wrapper

class AzureFunctionRouteWrapper:
    @staticmethod
    async def run_async(func, *args, **kwargs):
        return await func(*args, **kwargs)

    @staticmethod
    def run_sync(func, *args, **kwargs):
        return func(*args, **kwargs)