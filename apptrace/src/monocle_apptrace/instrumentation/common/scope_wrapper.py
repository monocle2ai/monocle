import logging
import inspect
from typing import Dict, List, Optional, Any
from functools import wraps
from contextlib import contextmanager, asynccontextmanager
from opentelemetry.context import Context
from monocle_apptrace.instrumentation.common.utils import (
    set_scope, remove_scope, http_route_handler, http_async_route_handler, set_scopes
)

logger = logging.getLogger(__name__)

def start_scope(
    scope_name: str, 
    scope_value: Optional[str] = None,
    context:  Optional[Context] = None
) -> object:
    """
    Start a new scope with the given name and optional value. If no value is provided, a random UUID will be generated.
    All the spans, across traces created after this call will have the scope attached until the scope is stopped.
    
    Args:
        scope_name: The name of the scope.
        scope_value: Optional value of the scope. If None, a random UUID will be generated.
    
    Returns:
        Token: A token representing the attached context for the scope. This token is to be used later to stop the current scope.
    """
    try:
        # Set the scope using existing utility
        token = set_scope(scope_name, scope_value, context)
        return token
    except Exception as e:
        logger.warning(f"Failed to start scope: {e}")
        return None

def stop_scope(
    token: object
) -> None:
    """
    Stop the active scope. All the spans created after this will not have the scope attached.
    
    Args:
        token: The token that was returned when the scope was started.
    
    Returns:
        None
    """
    try:
        # Remove the scope
        remove_scope(token)
    except Exception as e:
        logger.warning(f"Failed to stop scope: {e}")
    return

def start_scopes(
    scopes: Dict[str, Optional[str]],
    context:  Optional[Context] = None
) -> object:
    """
    Start multiple scopes with the given names and optional values. If no value is provided for a scope, a random UUID will be generated.
    All the spans, across traces created after this call will have the scopes attached until they are stopped.
    
    Args:
        scopes: A dictionary where keys are scope names and values are optional scope values. If a value is None, a random UUID will be generated.
    
    Returns:
        A dictionary where keys are scope names and values are tokens representing the attached contexts for the scopes. These tokens are to be used later to stop the respective scopes.
    """
    token = set_scopes(scopes, context)
    return token

@contextmanager
def monocle_trace_scope(
    scope_name: str, 
    scope_value: Optional[str] = None
):
    """
    Context manager to start and stop a scope. All the spans, across traces created within the encapsulated code will have the scope attached.
    
    Args:
        scope_name: The name of the scope.
        scope_value: Optional value of the scope. If None, a random UUID will be generated.
    """
    token = None
    if scope_name:
        token = start_scope(scope_name, scope_value)
    try:
        yield
    finally:
        stop_scope(token)

@asynccontextmanager
async def amonocle_trace_scope(
    scope_name: str, 
    scope_value: Optional[str] = None
):
    """
    Async context manager to start and stop a scope. All the spans, across traces created within the encapsulated code will have the scope attached.
    
    Args:
        scope_name: The name of the scope.
        scope_value: Optional value of the scope. If None, a random UUID will be generated.
    """
    token = start_scope(scope_name, scope_value)
    try:
        yield
    finally:
        stop_scope(token)
    
def monocle_trace_scope_method(
    scope_name: str, 
    scope_value: Optional[str] = None
):
    """
    Decorator to start and stop a scope for a method. All the spans, across traces created in the method will have the scope attached.
    
    Args:
        scope_name: The name of the scope.
        scope_value: Optional value of the scope. If None, a random UUID will be generated.
    """
    def decorator(func):
        if inspect.iscoroutinefunction(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                async with amonocle_trace_scope(
                    scope_name, scope_value
                ):
                    result = await func(*args, **kwargs)
                    return result
            return wrapper
        else:
            @wraps(func)
            def wrapper(*args, **kwargs):
                with monocle_trace_scope(
                    scope_name, scope_value
                ):
                    result = func(*args, **kwargs)
                    return result
            return wrapper
    return decorator
