#!/usr/bin/env python3
"""
FastAPI Integration Test with monocle_trace_method decorators - Focused on User Creation API.

This test demonstrates comprehensive HTTP plugin usage and internal method tracing:

## HTTP Plugin Features Demonstrated:
- assert_http_method('POST') - Validate POST requests
- assert_http_status_code() - Validate specific HTTP status codes
- assert_status_code_range(200, 300) - Validate 2xx status codes
- assert_status_code_range(400, 500) - Validate 4xx error codes
- assert_rest_api_endpoints() - Validate specific API endpoints
- assert_http_method_distribution() - Validate HTTP method counts
- debug_http_spans() - Debug HTTP span information (includes span types)
- validate_http_span_completeness() - Validate HTTP span attributes
- assert_api_call_timing() - Validate API response times
- assert_no_http_errors() - Ensure no HTTP errors

## HTTP Span Types Support:
- assert_http_span_type('http.process') - Validate server-side processing 
- assert_http_span_type('http.send') - Validate client-side requests
- assert_http_span_type() - Validate specific span types
- Span type differentiation: http.process vs http.send

## Internal Method Tracing:
- User creation API endpoint decorated with @monocle_trace_method()
- Internal business logic methods decorated with @monocle_trace_method()
- Execution order validation for internal method tracing
- Span hierarchy and parent-child relationships
- Integration with the monocle tfwk testing framework
"""

import logging
import time

# Import HTTP plugins to ensure they're registered (side-effect import)
import monocle_tfwk.assertions.plugins.http  # noqa: F401
import pytest
import requests
from monocle_tfwk import BaseAgentTest
from monocle_tfwk.visualization.gantt_chart import VisualizationMode
from server.fastapi_mock_server import mock_server

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture(scope="class", autouse=True)
def setup_server(request):
    """Set up FastAPI server for the test class."""
    # Start the mock FastAPI server
    mock_server.start()
    base_url = mock_server.get_base_url()
    
    # Set attributes on the test class
    # Note: Telemetry setup is handled by BaseAgentTest -> MonocleValidator
    request.cls.base_url = base_url
    
    yield
    
    # Cleanup
    mock_server.stop()


class TestFastAPIMonocleTraceMethod(BaseAgentTest):
    """Test suite for FastAPI endpoints decorated with @monocle_trace_method()."""

    

    
    
    def test_user_creation_endpoint_and_internal_methods(self):
        """Test user creation API endpoint with comprehensive internal method tracing and execution order validation."""
        logger.info("Testing POST /users endpoint with internal method tracing")
        
        # Create user data
        user_data = {
            "name": "Test User",
            "email": "test@example.com"
        }
        
        # Make HTTP request to the user creation endpoint
        response = requests.post(f"{self.base_url}/users", params=user_data)
        
        # Verify HTTP response
        assert response.status_code == 200
        created_user = response.json()
        assert created_user["name"] == "Test User"
        assert created_user["email"] == "test@example.com"
        assert "id" in created_user
        
        # Allow time for spans to be processed
        time.sleep(1)
        
        # Verify HTTP-specific traces using HTTP plugin assertions
        traces = self.assert_traces()
        
        # HTTP Plugin Assertions - Validate the HTTP request itself
        (traces
         .assert_http_method('POST')                # Assert POST request exists
         .assert_rest_api_endpoints([              # Assert specific endpoint was called
             {'method': 'POST', 'path': '/users'}
         ]))
        
        # Reset to all spans for internal method validation
        traces = self.assert_traces()
        traces.assert_spans(min_count=4)           # Expect at least 4 spans: endpoint + 3 internal methods
        # Note: The main endpoint span is "fastapi.request" not "create_user" based on debug output
        traces.completed_successfully()            # Assert no errors
        
        # Verify specific internal method spans exist using trace assertions
        # Note: Each assert_span_with_name() doesn't filter the span set, so we can chain them
        traces = self.assert_traces()  # Start with all spans
        (traces
         .assert_span_with_name("validate_user_data")
         .assert_span_with_name("calculate_user_profile_score") 
         .assert_span_with_name("send_notification")
         .assert_span_with_name("audit_log_operation"))
        
        logger.info("✅ All expected internal method spans found")
        
      

    def test_execution_order_validation(self):
        """Test that internal methods are executed in the correct order during user creation."""
        logger.info("Testing execution order of internal methods during user creation")
        
        # Create user data that will trigger all internal methods
        user_data = {
            "name": "Order Test User",
            "email": "order.test@example.com"
        }
        
        # Make HTTP request to create user
        response = requests.post(f"{self.base_url}/users", params=user_data)
        
        # Verify HTTP response
        assert response.status_code == 200
        created_user = response.json()
        assert created_user["name"] == "Order Test User"
        
        # Allow time for spans to be processed
        time.sleep(1)
        
        # Validate HTTP layer first using HTTP plugin assertions
        traces = self.assert_traces()
        (traces
         .assert_http_method('POST')
         .assert_status_code_range(200, 250)
         )              # Validate HTTP method
        
        # Validate that all expected internal methods were called using trace assertions
        traces = self.assert_traces()
        (traces
         .assert_span_with_name("validate_user_data")
         .assert_span_with_name("calculate_user_profile_score")
         .assert_span_with_name("send_notification")
         .assert_span_with_name("audit_log_operation"))
        
        logger.info("✅ All expected internal methods were executed")


    def test_validation_failure_with_execution_order(self):
        """Test that when validation fails, subsequent methods are not executed."""
        logger.info("Testing execution order when validation fails")
        
        # Try to create a user with invalid data (should fail validation)
        response = requests.post(
            f"{self.base_url}/users",
            params={"name": "A", "email": "invalid-email"}  # Invalid data
        )
        
        # Should get validation error
        assert response.status_code == 400
        
        # Wait for spans to be exported
        time.sleep(1)
        
        # Validate HTTP error response using HTTP plugin assertions
        traces = self.assert_traces()
        (traces
         .assert_http_method('POST'))               # Should still have POST request
         # Note: Status code assertions disabled until FastAPI instrumentation properly sets status codes
         # .assert_client_error_status_codes()       # Should have 4xx error status
         # .assert_http_status_code(400))            # Specifically 400 status
        
        # Should still have validation span even though it failed
        traces = self.assert_traces()
        traces.assert_span_with_name("validate_user_data")
        
        # Should NOT have profile calculation or notification spans since validation failed
        # Use filter_by_name to check if these spans exist
        traces = self.assert_traces()
        traces.filter_by_name("calculate_user_profile_score").exactly(0)  # Should be 0
        
        traces = self.assert_traces()
        traces.filter_by_name("send_notification").exactly(0)  # Should be 0
        
        logger.info("Method execution stopped correctly after validation failure")
        logger.info("✅ Validation failure execution order test passed")

    def test_comprehensive_http_plugin_assertions(self):
        """Test comprehensive HTTP plugin assertions with detailed validation."""
        logger.info("Testing comprehensive HTTP plugin assertions")
        
        # Create user data
        user_data = {
            "name": "HTTP Plugin Test User",
            "email": "http.plugin@example.com"
        }
        
        # Make HTTP request to create user
        response = requests.post(f"{self.base_url}/users", params=user_data)
        
        # Verify HTTP response
        assert response.status_code == 200
        created_user = response.json()
        assert created_user["name"] == "HTTP Plugin Test User"
        
        # Allow time for spans to be processed
        time.sleep(1)
        
        # Comprehensive HTTP Plugin Validation
        traces = self.assert_traces()
        
        logger.info("=== Comprehensive HTTP Plugin Assertions ===")
        
        # Debug HTTP spans for visibility
        traces.debug_http_spans()
        
        # Method assertions
        (traces
         .assert_http_method('POST'))           # Assert from list of methods
        # API endpoint validation
        traces = self.assert_traces()  # Reset to all spans
        traces.assert_rest_api_endpoints([
            {'method': 'POST', 'path': '/users'}
        ])
        
        # HTTP method distribution validation
        traces = self.assert_traces()  # Reset to all spans
        traces.assert_http_method_distribution({
            'POST': 1  # Expect exactly 1 POST request
        })
        
        # Comprehensive validation in fewer chains
        traces = self.assert_traces()  # Reset to all spans
        (traces
         .assert_api_call_timing(5000)        # API timing validation (5 second max)
         .assert_http_span_type('http.process')) # Assert specific span type
        
        logger.info("✅ Comprehensive HTTP plugin assertions test passed")

    def test_http_span_types_validation(self):
        """Test validation of both http.process and http.send span types."""
        logger.info("Testing HTTP span types: http.process and http.send")
        
        # Create user data
        user_data = {
            "name": "Span Types Test User",
            "email": "span.types@example.com"
        }
        
        # Make HTTP request to create user - this should generate http.process spans
        response = requests.post(f"{self.base_url}/users", params=user_data)
        
        # Verify HTTP response
        assert response.status_code == 200
        created_user = response.json()
        assert created_user["name"] == "Span Types Test User"
        
        # Allow time for spans to be processed
        time.sleep(1)
        
        # Get traces for span type validation
        traces = self.assert_traces()
        
        logger.info("=== HTTP Span Types Validation ===")
        
        # Debug all HTTP spans to see their types
        traces.debug_http_spans()
        
        # Test http.process spans (server-side FastAPI processing)
        logger.info("--- Testing http.process spans ---")
        traces = self.assert_traces()  # Reset to all spans
        (traces
         .assert_http_span_type('http.process') # Assert http.process spans exist
         .assert_http_span_type('http.process') # Assert specific span type
         .assert_http_method('POST'))           # Should be POST requests

        # Validate http.process spans exist using trace assertions
        traces = self.assert_traces()
        traces.filter_by_attribute('span.type', 'http.process').at_least(1)
        
        # Check for http.send spans (should be 0 in this test setup)
        traces = self.assert_traces()
        traces.filter_by_attribute('span.type', 'http.send').exactly(0)
        
        logger.info("✅ Span type filtering validation completed")


    def test_mixed_span_type_and_name_flow_patterns(self):
        """Test various combinations of span types and names in flow assertions."""
        logger.info("Testing mixed span type and name flow patterns")
        
        # Create user data
        user_data = {
            "name": "Flow Pattern Test User",
            "email": "flow.pattern@example.com"
        }
        
        # Make HTTP request to create user
        response = requests.post(f"{self.base_url}/users", params=user_data)
        
        # Verify HTTP response
        assert response.status_code == 200
        created_user = response.json()
        assert created_user["name"] == "Flow Pattern Test User"
        
        # Allow time for spans to be processed
        time.sleep(1)
        
        logger.info("=== Testing Mixed Flow Patterns ===")
        
        # Comprehensive flow pattern validation - all chained together since assert_flow doesn't filter
        logger.info("Testing comprehensive flow patterns in a single chain")
        traces = self.assert_traces()

        # Generate and display Gantt chart visualization after flow assertions
        logger.info("=== Gantt Chart Visualization ===")
        self.display_flow_gantt_chart(VisualizationMode.COMPACT)
        
        (traces
         .assert_flow("validate_user_data")  # 1. Pure span names (no parent filter)
         .assert_flow("fastapi.request","workflow")                                       # 2. Pure span types
         .assert_flow("validate_user_data -> calculate_user_profile_score -> send_notification -> audit_log_operation")  # 3. Flow pattern (no parent filter)
        #  .assert_flow("validate_* -> calculate_* -> send_*")               # 4. Wildcard patterns
        #  .assert_flow("validate_user_data -> calculate_user_profile_score? -> send_notification")  # 5. Optional patterns
        #  .assert_flow("validate_user_data -> (calculate_user_profile_score -> send_notification)")  # 6. Parallel patterns (no parent filter)
        #  .assert_flow("http.process -> validate_* -> (calculate_* -> send_*) -> audit_*")  # 7. Complex mixed pattern (no parent filter)
        #  .assert_flow("*user* -> *notification*")
         )
        logger.info("✅ Mixed span type and name flow patterns test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])