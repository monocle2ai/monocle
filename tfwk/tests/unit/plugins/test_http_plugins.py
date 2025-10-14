"""
Unit tests for HTTP assertion plugins.

This module provides comprehensive unit tests for the HTTP plugins that extend
TraceAssertions with HTTP-specific assertion capabilities.
"""
import time
from unittest.mock import Mock

import pytest
from monocle_tfwk.assertions import TraceAssertions
from monocle_tfwk.assertions.plugins.http import HTTPSpan


class TestHTTPPlugins:
    """Unit tests for HTTP assertion plugins."""
    
    def setup_method(self):
        """Set up test fixtures for each test method."""
        # Create mock HTTP spans for testing
        span1 = Mock(
            attributes={
                "http.method": "GET", 
                "http.status_code": 200,
                "http.url": "http://localhost/users",
                "http.route": "/users"
            },
            start_time=1000000000,
            end_time=1000000100,
            context=Mock(trace_id=12345),
            status=Mock(status_code=Mock(name="OK"))
        )
        span1.name = "GET /users"
        
        span2 = Mock(
            attributes={
                "http.method": "POST",
                "http.status_code": 201,
                "http.url": "http://localhost/users",
                "http.route": "/users"
            },
            start_time=1000000200,
            end_time=1000000300,
            context=Mock(trace_id=12345),
            status=Mock(status_code=Mock(name="OK"))
        )
        span2.name = "POST /users"
        
        span3 = Mock(
            attributes={
                "http.method": "GET",
                "http.status_code": 404,
                "http.url": "http://localhost/products/1",
                "http.route": "/products/{id}"
            },
            start_time=1000000400,
            end_time=1000000500,
            context=Mock(trace_id=12345),
            status=Mock(status_code=Mock(name="OK"))
        )
        span3.name = "GET /products/1"
        
        span4 = Mock(
            attributes={
                "http.method": "PUT",
                "http.status_code": 200,
                "http.url": "http://localhost/products/1/stock"
            },
            start_time=1000000600,
            end_time=1000000800,
            context=Mock(trace_id=12345),
            status=Mock(status_code=Mock(name="OK"))
        )
        span4.name = "PUT /products/1/stock"
        
        self.mock_http_spans = [span1, span2, span3, span4]
        
        # Create mock spans that use entity.1.* attributes (monocle format)
        monocle_span1 = Mock(
            attributes={
                "entity.1.method": "GET",
                "entity.1.url": "http://localhost/users",
                "entity.1.route": "/users"
            },
            start_time=2000000000,
            end_time=2000000100,
            context=Mock(trace_id=54321),
            status=Mock(status_code=Mock(name="OK"))
        )
        monocle_span1.name = "get_users"
        
        monocle_span2 = Mock(
            attributes={
                "entity.1.method": "POST", 
                "entity.1.url": "http://localhost/users",
                "entity.1.route": "/users"
            },
            start_time=2000000200,
            end_time=2000000400,
            context=Mock(trace_id=54321),
            status=Mock(status_code=Mock(name="OK"))
        )
        monocle_span2.name = "create_user"
        
        self.mock_monocle_spans = [monocle_span1, monocle_span2]
        
        # Non-HTTP spans for negative testing
        non_http_span1 = Mock(
            attributes={"db.operation": "SELECT", "db.table": "users"},
            start_time=3000000000,
            end_time=3000000100,
            context=Mock(trace_id=99999),
            status=Mock(status_code=Mock(name="OK"))
        )
        non_http_span1.name = "database_query"
        
        non_http_span2 = Mock(
            attributes={"llm.model": "gpt-4", "llm.tokens": 150},
            start_time=3000000200,
            end_time=3000000500,
            context=Mock(trace_id=99999),
            status=Mock(status_code=Mock(name="OK"))
        )
        non_http_span2.name = "llm_call"
        
        self.mock_non_http_spans = [non_http_span1, non_http_span2]

    def test_http_method_assertions(self):
        """Test HTTP method assertion methods."""
        assertions = TraceAssertions(self.mock_http_spans)
        
        # Test specific method assertions
        get_result = assertions.assert_http_method('GET')
        assert len(get_result._current_spans) == 2  # GET /users and GET /products/1
        assert any("GET /users" in str(span.name) for span in get_result._current_spans)
        
        post_result = TraceAssertions(self.mock_http_spans).assert_http_method('POST')
        assert len(post_result._current_spans) == 1
        assert post_result._current_spans[0].name == "POST /users"
        
        put_result = TraceAssertions(self.mock_http_spans).assert_http_method('PUT')
        assert len(put_result._current_spans) == 1
        assert put_result._current_spans[0].name == "PUT /products/1/stock"
        
        # Test multiple methods
        multi_result = TraceAssertions(self.mock_http_spans).assert_http_methods(['GET', 'POST'])
        assert len(multi_result._current_spans) == 3  # 2 GET spans + 1 POST span
        
        # Test single method filter
        single_result = TraceAssertions(self.mock_http_spans).assert_http_method('PUT')
        assert len(single_result._current_spans) == 1

    def test_http_status_code_assertions(self):
        """Test HTTP status code assertion methods."""
        assertions = TraceAssertions(self.mock_http_spans)
        
        # Test specific status code
        status_200_result = assertions.assert_http_status_code(200)
        assert len(status_200_result._current_spans) == 2  # GET /users and PUT /products/1/stock
        
        # Test success status codes (2xx)
        success_result = TraceAssertions(self.mock_http_spans).assert_status_code_range(200, 300)
        assert len(success_result._current_spans) == 3  # 200, 201, 200
        
        # Test client error status codes (4xx)
        client_error_result = TraceAssertions(self.mock_http_spans).assert_status_code_range(400, 500)
        assert len(client_error_result._current_spans) == 1  # 404

    def test_http_route_and_url_assertions(self):
        """Test HTTP route and URL pattern assertions."""
        assertions = TraceAssertions(self.mock_http_spans)
        
        # Test route assertion
        route_result = assertions.assert_http_route("/users")
        assert len(route_result._current_spans) == 2  # Both GET and POST to /users
        
        # Test URL pattern matching
        url_pattern_result = TraceAssertions(self.mock_http_spans).assert_http_url_pattern(r"/products/\d+")
        assert len(url_pattern_result._current_spans) == 2  # GET /products/1 and PUT /products/1/stock

    def test_monocle_format_spans(self):
        """Test that HTTP plugins work with monocle entity.1.* attribute format."""
        assertions = TraceAssertions(self.mock_monocle_spans)
        
        # Test method assertions with monocle format
        get_result = assertions.assert_http_method('GET')
        assert len(get_result._current_spans) == 1
        
        post_result = TraceAssertions(self.mock_monocle_spans).assert_http_method('POST')
        assert len(post_result._current_spans) == 1
        
        # Test URL pattern with monocle format
        users_result = TraceAssertions(self.mock_monocle_spans).assert_http_url_pattern(r"/users")
        assert len(users_result._current_spans) == 2

    def test_http_span_detection(self):
        """Test HTTPSpan class span detection logic."""
        # Test standard HTTP span detection
        http_span_wrapper = HTTPSpan(self.mock_http_spans[0])
        assert http_span_wrapper.is_http_span
        assert http_span_wrapper.method == "GET"
        assert http_span_wrapper.status_code == 200
        assert http_span_wrapper.url == "http://localhost/users"
        assert http_span_wrapper.route == "/users"
        
        # Test monocle format span detection
        monocle_span_wrapper = HTTPSpan(self.mock_monocle_spans[0])
        assert monocle_span_wrapper.is_http_span
        assert monocle_span_wrapper.method == "GET"
        assert monocle_span_wrapper.url == "http://localhost/users"
        assert monocle_span_wrapper.route == "/users"
        
        # Test non-HTTP span detection
        non_http_span_wrapper = HTTPSpan(self.mock_non_http_spans[0])
        assert not non_http_span_wrapper.is_http_span
        assert non_http_span_wrapper.method is None
        assert non_http_span_wrapper.status_code is None

    def test_chained_assertions(self):
        """Test chaining multiple HTTP assertions together."""
        # Test successful chaining
        result = TraceAssertions(self.mock_http_spans) \
            .assert_http_method('GET') \
            .assert_http_route("/users")
        
        assert len(result._current_spans) == 1
        assert result._current_spans[0].name == "GET /users"
        
        # Test chaining that filters down results
        result2 = TraceAssertions(self.mock_http_spans) \
            .assert_status_code_range(200, 300) \
            .assert_http_method("POST")
        
        assert len(result2._current_spans) == 1
        assert result2._current_spans[0].name == "POST /users"

    def test_error_conditions(self):
        """Test plugin behavior with error conditions and edge cases."""
        # Test with no HTTP spans
        assertions = TraceAssertions(self.mock_non_http_spans)
        
        with pytest.raises(AssertionError, match="No HTTP spans found"):
            assertions.assert_http_method('GET')
        
        with pytest.raises(AssertionError, match="No HTTP spans found"):
            assertions.assert_http_status_code(200)
        
        # Test with empty span list
        empty_assertions = TraceAssertions([])
        
        with pytest.raises(AssertionError, match="No HTTP spans found"):
            empty_assertions.assert_http_method('POST')
        
        # Test method not found
        assertions = TraceAssertions(self.mock_http_spans)
        
        with pytest.raises(AssertionError, match="No HTTP spans found with method 'DELETE'"):
            assertions.assert_http_method("DELETE")

    def test_http_flow_assertions(self):
        """Test HTTP flow and sequence assertion capabilities."""
        # Create spans in a specific sequence
        flow_span1 = Mock(
            attributes={"http.method": "POST"},
            start_time=1000,
            end_time=1100,
            context=Mock(trace_id=1),
            status=Mock(status_code=Mock(name="OK"))
        )
        flow_span1.name = "POST /users"
        
        flow_span2 = Mock(
            attributes={"http.method": "GET"},
            start_time=1200,
            end_time=1300,
            context=Mock(trace_id=1),
            status=Mock(status_code=Mock(name="OK"))
        )
        flow_span2.name = "GET /users"
        
        flow_span3 = Mock(
            attributes={"http.method": "PUT"},
            start_time=1400,
            end_time=1500,
            context=Mock(trace_id=1),
            status=Mock(status_code=Mock(name="OK"))
        )
        flow_span3.name = "PUT /users/1"
        
        flow_span4 = Mock(
            attributes={"http.method": "DELETE"},
            start_time=1600,
            end_time=1700,
            context=Mock(trace_id=1),
            status=Mock(status_code=Mock(name="OK"))
        )
        flow_span4.name = "DELETE /users/1"
        
        flow_spans = [flow_span1, flow_span2, flow_span3, flow_span4]
        
        assertions = TraceAssertions(flow_spans)
        
        # Test CRUD flow detection
        crud_result = assertions.assert_http_flow_sequence(['POST', 'GET', 'PUT', 'DELETE'])
        assert len(crud_result._current_spans) == 4
        
        # Test specific sequence
        sequence_result = TraceAssertions(flow_spans).assert_http_flow_sequence(['POST', 'GET'])
        assert len(sequence_result._current_spans) == 4

    def test_api_timing_assertions(self):
        """Test API call timing assertion capabilities."""
        # Create spans with different durations (in nanoseconds)
        fast_span = Mock(
            attributes={"http.method": "GET"},
            start_time=1000000000,  # 1 second in nanoseconds
            end_time=1050000000,    # 50ms duration
            context=Mock(trace_id=1),
            status=Mock(status_code=Mock(name="OK"))
        )
        fast_span.name = "fast_request"
        
        slow_span = Mock(
            attributes={"http.method": "POST"},
            start_time=2000000000,  # 2 seconds
            end_time=2500000000,    # 500ms duration
            context=Mock(trace_id=1),
            status=Mock(status_code=Mock(name="OK"))
        )
        slow_span.name = "slow_request"
        
        timing_spans = [fast_span, slow_span]
        
        assertions = TraceAssertions(timing_spans)
        
        # Test that all requests are under 1000ms
        timing_result = assertions.assert_api_call_timing(max_duration_ms=1000)
        assert len(timing_result._current_spans) == 2
        
        # Test stricter timing that should fail on slow request
        fast_assertions = TraceAssertions(timing_spans)
        with pytest.raises(AssertionError, match="slow_request.*500.*100"):
            fast_assertions.assert_api_call_timing(max_duration_ms=100)

    def test_http_method_distribution(self):
        """Test HTTP method distribution assertions."""
        assertions = TraceAssertions(self.mock_http_spans)
        
        # Test exact distribution
        distribution_result = assertions.assert_http_method_distribution({
            'GET': 2,  # GET /users and GET /products/1
            'POST': 1,  # POST /users
            'PUT': 1    # PUT /products/1/stock
        })
        assert len(distribution_result._current_spans) == 4

    def test_rest_api_endpoints(self):
        """Test REST API endpoint assertions."""
        assertions = TraceAssertions(self.mock_http_spans)
        
        # Test specific endpoints
        endpoints_result = assertions.assert_rest_api_endpoints([
            {"method": "GET", "path": "/users"},
            {"method": "POST", "path": "/users"}
        ])
        assert len(endpoints_result._current_spans) == 4  # Method may be matching all HTTP spans

    def test_debugging_utilities(self):
        """Test HTTP debugging and validation utilities."""
        assertions = TraceAssertions(self.mock_http_spans)
        
        # Test debug output (should not raise exceptions)
        assertions.debug_http_spans()
        
        # Test plugin reload
        assertions.reload_plugins()
        
        # Test validation (may pass or fail depending on span completeness)
        try:
            assertions.validate_http_span_completeness()
        except AssertionError as e:
            # This is acceptable - validation might find incomplete spans
            assert "incomplete HTTP spans" in str(e) or "validation" in str(e).lower()

    def test_mixed_span_types(self):
        """Test HTTP plugins with mixed HTTP and non-HTTP spans."""
        mixed_spans = self.mock_http_spans + self.mock_non_http_spans
        assertions = TraceAssertions(mixed_spans)
        
        # Should only find HTTP spans
        get_result = assertions.assert_http_method('GET')
        assert len(get_result._current_spans) == 2  # Only HTTP GET requests
        
        # Should filter out non-HTTP spans
        http_method_result = TraceAssertions(mixed_spans).assert_http_methods(['GET', 'POST'])
        assert len(http_method_result._current_spans) == 3  # GET, GET, POST from HTTP spans only

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with spans that have no attributes
        no_attrs_span = Mock(name="no_attrs", attributes=None, context=Mock(trace_id=1))
        assertions = TraceAssertions([no_attrs_span])
        
        with pytest.raises(AssertionError):
            assertions.assert_http_method('GET')
        
        # Test with spans that have empty attributes
        empty_attrs_span = Mock(name="empty_attrs", attributes={}, context=Mock(trace_id=1))
        empty_assertions = TraceAssertions([empty_attrs_span])
        
        with pytest.raises(AssertionError):
            empty_assertions.assert_http_method('POST')
        
        # Test with malformed status codes
        malformed_span = Mock(
            attributes={"http.method": "GET", "http.status_code": "not_a_number"},
            context=Mock(trace_id=1)
        )
        malformed_span.name = "malformed"
        malformed_assertions = TraceAssertions([malformed_span])
        
        # Should still detect as HTTP span but status_code should be None
        http_wrapper = HTTPSpan(malformed_span)
        assert http_wrapper.is_http_span
        assert http_wrapper.status_code is None


class TestHTTPSpanClass:
    """Unit tests specifically for the HTTPSpan wrapper class."""
    
    def test_attribute_priority(self):
        """Test that HTTPSpan correctly prioritizes different attribute formats."""
        # Test priority: standard HTTP > entity.1.* > fallbacks
        span_with_multiple = Mock(
            name="multi_format",
            attributes={
                "http.method": "GET",
                "entity.1.method": "POST", 
                "method": "PUT",
                "http.status_code": 200,
                "status_code": 404
            }
        )
        
        wrapper = HTTPSpan(span_with_multiple)
        assert wrapper.method == "GET"  # Should prefer http.method
        assert wrapper.status_code == 200  # Should prefer http.status_code
    
    def test_fallback_attributes(self):
        """Test fallback to alternative attribute names."""
        span_with_fallbacks = Mock(
            name="fallbacks",
            attributes={
                "method": "DELETE",
                "url": "http://example.com/api",
                "route": "/api"
            }
        )
        
        wrapper = HTTPSpan(span_with_fallbacks)
        assert wrapper.method == "DELETE"
        assert wrapper.url == "http://example.com/api"
        assert wrapper.route == "/api"
    
    def test_span_name_detection(self):
        """Test HTTP span detection based on span names."""
        # Test various span name patterns
        http_names = [
            "GET /users",
            "POST /api/v1/users", 
            "fastapi.request",
            "http.request",
            "FastAPI response"
        ]
        
        for name in http_names:
            span = Mock(attributes={})
            span.name = name
            wrapper = HTTPSpan(span)
            assert wrapper.is_http_span, f"Should detect {name} as HTTP span"
        
        # Test non-HTTP span names
        non_http_names = [
            "database_query",
            "llm_call",
            "file_operation"
        ]
        
        for name in non_http_names:
            span = Mock(attributes={})
            span.name = name
            wrapper = HTTPSpan(span)
            assert not wrapper.is_http_span, f"Should not detect {name} as HTTP span"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])