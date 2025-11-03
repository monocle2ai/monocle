"""
Common pytest fixtures for all test modules.
"""
import os
from contextlib import contextmanager

import pytest


@pytest.fixture
def preserve_env():
    """Fixture to preserve and restore environment variables."""
    original_env = os.environ.copy()
    yield
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@contextmanager
def temporary_env_var(key, value):
    """Context manager to temporarily set an environment variable."""
    original_value = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if original_value is not None:
            os.environ[key] = original_value
        elif key in os.environ:
            del os.environ[key]


@contextmanager
def temporary_env_vars(**env_vars):
    """Context manager to temporarily set multiple environment variables."""
    original_values = {}
    for key, value in env_vars.items():
        original_values[key] = os.environ.get(key)
        os.environ[key] = value
    
    try:
        yield
    finally:
        for key, original_value in original_values.items():
            if original_value is not None:
                os.environ[key] = original_value
            elif key in os.environ:
                del os.environ[key]