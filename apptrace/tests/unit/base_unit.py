"""
Base test class providing standardized setup and teardown for all unit tests.
This ensures proper instrumentation cleanup and consistent test isolation.
"""

import logging
import os
import unittest
import warnings

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import set_tracer_provider


class MonocleTestBase(unittest.TestCase):
    """
    Base test class that provides standardized setup and teardown methods
    for proper instrumentation state management and test isolation.
    """
    
    instrumentor = None
    
    def setUp(self):
        """
        Standard setup method that should be called by all test classes.
        Ensures clean environment and proper instrumentation initialization.
        """
        # Set up clean environment variables
        os.environ["HTTP_API_KEY"] = "test-key-123"
        os.environ["HTTP_INGESTION_ENDPOINT"] = "https://localhost:3000/api/v1/traces"
        
        # Reset OpenTelemetry state - create fresh tracer provider
        tracer_provider = TracerProvider()
        set_tracer_provider(tracer_provider)
        
        # Clear any cached instrumentation state
        self._clear_instrumentation_cache()
        
        # Suppress noisy warnings for cleaner test output
        self._suppress_test_warnings()
        
        # Initialize instrumentor (to be overridden by subclasses)
        self.instrumentor = None
    
    def tearDown(self):
        """
        Standard teardown method that ensures proper cleanup of instrumentation state.
        This should be called by all test classes to prevent state leakage between tests.
        """
        # Clean up instrumentation
        if self.instrumentor is not None:
            try:
                self.instrumentor.uninstrument()
            except Exception as e:
                logging.warning(f"Uninstrument failed: {e}")
        
        # Clean up any global state that might affect other tests
        import gc
        gc.collect()
        
        # Reset warning filters
        warnings.resetwarnings()
        
        # Clear any monocle instrumentation state
        self._clear_monocle_state()
    
    def _suppress_test_warnings(self):
        """Suppress noisy warnings that clutter test output"""
        # Suppress instrumentation warnings
        logging.getLogger('monocle_apptrace.instrumentation.common.instrumentor').setLevel(logging.CRITICAL)
        logging.getLogger('google.adk.models.registry').setLevel(logging.CRITICAL)
        logging.getLogger('haystack.components.builders.prompt_builder').setLevel(logging.CRITICAL)
        logging.getLogger('opentelemetry.attributes').setLevel(logging.CRITICAL)
        logging.getLogger('tests.common.http_span_exporter').setLevel(logging.CRITICAL)        
        # Suppress pydantic warnings
        warnings.filterwarnings("ignore", category=UserWarning, module="pydantic._internal._fields")
    
    def _clear_instrumentation_cache(self):
        """Clear any cached instrumentation state that might interfere with tests"""
        try:
            import sys
            
            # Clear any modules that might have cached state
            modules_to_clear = [
                'monocle_apptrace.instrumentation.common.instrumentor',
                'monocle_apptrace.instrumentation.common.span_handler',
                'monocle_apptrace.instrumentation.common.wrapper'
            ]
            
            for module_name in modules_to_clear:
                if module_name in sys.modules:
                    module = sys.modules[module_name]
                    # Clear any __dict__ caches if they exist
                    if hasattr(module, '__dict__'):
                        # Reset specific caches that might exist
                        cache_attrs = ['_cached_spans', '_instrumented_methods', '_wrapper_cache']
                        for attr in cache_attrs:
                            if hasattr(module, attr):
                                setattr(module, attr, {})
        except Exception:
            # If clearing cache fails, continue - it's not critical
            pass
    
    def _clear_monocle_state(self):
        """Clear any monocle instrumentation state"""
        try:
            # Try to reset any global instrumentor state if available
            from monocle_apptrace.instrumentation.common.instrumentor import (
                reset_telemetry,
            )
            reset_telemetry()
        except (ImportError, AttributeError):
            # If reset function doesn't exist, that's fine
            pass


class MonocleAsyncTestBase(unittest.IsolatedAsyncioTestCase):
    """
    Base async test class that provides standardized setup and teardown methods
    for async test cases with proper instrumentation state management.
    """
    
    instrumentor = None
    
    def setUp(self):
        """Standard async setup method"""
        # Set up clean environment variables
        os.environ["HTTP_API_KEY"] = "test-key-123"
        os.environ["HTTP_INGESTION_ENDPOINT"] = "https://localhost:3000/api/v1/traces"
        
        # Reset OpenTelemetry state - create fresh tracer provider
        tracer_provider = TracerProvider()
        set_tracer_provider(tracer_provider)
        
        # Clear any cached instrumentation state
        self._clear_instrumentation_cache()
        
        # Suppress noisy warnings for cleaner test output
        self._suppress_test_warnings()
        
        # Initialize instrumentor (to be overridden by subclasses)
        self.instrumentor = None
    
    def tearDown(self):
        """Standard async teardown method"""
        # Clean up instrumentation
        if self.instrumentor is not None:
            try:
                self.instrumentor.uninstrument()
            except Exception as e:
                logging.warning(f"Uninstrument failed: {e}")
        
        # Clean up any global state that might affect other tests
        import gc
        gc.collect()
        
        # Reset warning filters
        warnings.resetwarnings()
        
        # Clear any monocle instrumentation state
        self._clear_monocle_state()
    
    def _suppress_test_warnings(self):
        """Suppress noisy warnings that clutter test output"""
        # Suppress instrumentation warnings
        logging.getLogger('monocle_apptrace.instrumentation.common.instrumentor').setLevel(logging.CRITICAL)
        logging.getLogger('google.adk.models.registry').setLevel(logging.CRITICAL)
        logging.getLogger('haystack.components.builders.prompt_builder').setLevel(logging.CRITICAL)
        logging.getLogger('opentelemetry.attributes').setLevel(logging.CRITICAL)
        logging.getLogger('tests.common.http_span_exporter').setLevel(logging.CRITICAL)        
        # Suppress pydantic warnings
        warnings.filterwarnings("ignore", category=UserWarning, module="pydantic._internal._fields")
    
    def _clear_instrumentation_cache(self):
        """Clear any cached instrumentation state that might interfere with tests"""
        try:
            import sys
            
            # Clear any modules that might have cached state
            modules_to_clear = [
                'monocle_apptrace.instrumentation.common.instrumentor',
                'monocle_apptrace.instrumentation.common.span_handler',
                'monocle_apptrace.instrumentation.common.wrapper'
            ]
            
            for module_name in modules_to_clear:
                if module_name in sys.modules:
                    module = sys.modules[module_name]
                    # Clear any __dict__ caches if they exist
                    if hasattr(module, '__dict__'):
                        # Reset specific caches that might exist
                        cache_attrs = ['_cached_spans', '_instrumented_methods', '_wrapper_cache']
                        for attr in cache_attrs:
                            if hasattr(module, attr):
                                setattr(module, attr, {})
        except Exception:
            # If clearing cache fails, continue - it's not critical
            pass
    
    def _clear_monocle_state(self):
        """Clear any monocle instrumentation state"""
        try:
            # Try to reset any global instrumentor state if available
            from monocle_apptrace.instrumentation.common.instrumentor import (
                reset_telemetry,
            )
            reset_telemetry()
        except (ImportError, AttributeError):
            # If reset function doesn't exist, that's fine
            pass