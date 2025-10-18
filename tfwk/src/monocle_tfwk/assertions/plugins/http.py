"""
HTTP assertion plugins for web API and HTTP-related span validation.

This module provides assertion plugins specifically designed for validating
HTTP requests, responses, methods, status codes, and flow patterns in web API
testing scenarios.

Simplified API:
- Use assert_http_method('GET') instead of assert_get_requests()
- Use assert_status_code_range(200, 300) instead of assert_success_status_codes()
- Use assert_http_span_type('http.process') instead of assert_http_process_spans()
- Use assert_http_flow_sequence(['POST', 'GET', 'PUT', 'DELETE']) for CRUD flows
"""
import re
from typing import TYPE_CHECKING, Dict, List, Optional

from monocle_tfwk.assertions.plugin_registry import TraceAssertionsPlugin, plugin

if TYPE_CHECKING:
    from monocle_tfwk.assertions.trace_assertions import TraceAssertions


class HTTPSpan:
    """Wrapper class for HTTP-related span information."""
    
    # Pre-compiled regex patterns for entity attribute matching (class-level constants)
    ENTITY_METHOD_PATTERN = re.compile(r'^entity\.\d+\.method$')
    ENTITY_STATUS_CODE_PATTERN = re.compile(r'^entity\.\d+\.status_code$')
    ENTITY_URL_PATTERN = re.compile(r'^entity\.\d+\.url$')
    ENTITY_ROUTE_PATTERN = re.compile(r'^entity\.\d+\.route$')
    
    # Standard HTTP attribute sets for organized lookups
    HTTP_METHOD_ATTRS = {'http.method', 'http.request.method', 'method'}
    HTTP_STATUS_ATTRS = {'http.status_code', 'http.response.status_code', 'status_code'}
    HTTP_URL_ATTRS = {'http.url', 'http.target', 'url'}
    HTTP_ROUTE_ATTRS = {'http.route', 'route'}
    
    # HTTP span name indicators
    REQUEST_INDICATORS = {'request', 'fastapi.request', 'http.request', 'client'}
    RESPONSE_INDICATORS = {'response', 'fastapi.response', 'http.response'}
    HTTP_INDICATORS = {'http', 'fastapi', 'request', 'response', 'get_', 'post_', 'put_', 'delete_'}
    HTTP_METHOD_PREFIXES = ('get ', 'post ', 'put ', 'delete ', 'patch ', 'head ', 'options ')
    
    # Span type constants
    SPAN_TYPE_ATTR = 'span.type'
    HTTP_PROCESS_SPAN_TYPE = 'http.process'  # Server-side processing
    HTTP_SEND_SPAN_TYPE = 'http.send'        # Client-side requests
    
    # HTTP status code ranges
    CLIENT_ERROR_MIN = 400
    SERVER_ERROR_MIN = 500
    
    
    # Time conversion constants
    NANOSECONDS_TO_MILLISECONDS = 1_000_000
    
    def __init__(self, span):
        self.span = span
        self.name = span.name
        self.attributes = span.attributes or {}
    
    def _get_first_matching_attr(self, attr_set: set) -> Optional[str]:
        """Get the first matching attribute value from a set of possible attribute names."""
        for attr_name in attr_set:
            value = self.attributes.get(attr_name)
            if value:
                return value
        return None
    
    def _get_entity_pattern_value(self, pattern) -> Optional[str]:
        """Get value matching entity pattern from attributes."""
        for key, value in self.attributes.items():
            if pattern.match(key):
                return value
        return None
        
    @property
    def method(self) -> Optional[str]:
        """Get HTTP method from span attributes."""
        # Try standard HTTP method attributes first
        method = self._get_first_matching_attr(self.HTTP_METHOD_ATTRS)
        if method:
            return method
            
        # Check for entity.*.method pattern using pre-compiled regex
        return self._get_entity_pattern_value(self.ENTITY_METHOD_PATTERN)
    
    @property
    def status_code(self) -> Optional[int]:
        """Get HTTP status code from span attributes."""
        # Try standard HTTP status code attributes first
        status = self._get_first_matching_attr(self.HTTP_STATUS_ATTRS)
        
        if status is None:
            # Check for entity.*.status_code pattern using pre-compiled regex
            status = self._get_entity_pattern_value(self.ENTITY_STATUS_CODE_PATTERN)
                    
        if status is not None:
            try:
                return int(status)
            except ValueError:
                return None
        return None
    
    @property
    def url(self) -> Optional[str]:
        """Get URL from span attributes."""
        # Try standard URL attributes first
        url = self._get_first_matching_attr(self.HTTP_URL_ATTRS)
        if url:
            return url
            
        # Check for entity.*.url pattern using pre-compiled regex
        return self._get_entity_pattern_value(self.ENTITY_URL_PATTERN)
    
    @property
    def route(self) -> Optional[str]:
        """Get HTTP route from span attributes."""
        # Try standard route attributes first
        route = self._get_first_matching_attr(self.HTTP_ROUTE_ATTRS)
        if route:
            return route
            
        # Check for entity.*.route pattern using pre-compiled regex
        return self._get_entity_pattern_value(self.ENTITY_ROUTE_PATTERN)
    
    @property
    def is_request_span(self) -> bool:
        """Check if this is an HTTP request span."""
        name_lower = str(self.name).lower()
        return any(indicator in name_lower for indicator in self.REQUEST_INDICATORS)
    
    @property 
    def is_response_span(self) -> bool:
        """Check if this is an HTTP response span."""
        name_lower = str(self.name).lower()
        return any(indicator in name_lower for indicator in self.RESPONSE_INDICATORS)
    
    @property
    def span_type(self) -> Optional[str]:
        """Get span type from span attributes."""
        return self.attributes.get(self.SPAN_TYPE_ATTR)
    
    @property
    def is_http_process_span(self) -> bool:
        """Check if this is an HTTP server-side processing span (http.process)."""
        return self.span_type == self.HTTP_PROCESS_SPAN_TYPE
    
    @property
    def is_http_send_span(self) -> bool:
        """Check if this is an HTTP client-side request span (http.send)."""
        return self.span_type == self.HTTP_SEND_SPAN_TYPE
    
    @property
    def is_http_span(self) -> bool:
        """Check if this span is HTTP-related."""
        name_lower = str(self.name).lower()
        
        return (self.is_http_process_span or 
                self.is_http_send_span or
                self.method is not None or 
                self.status_code is not None or
                self.url is not None or
                any(indicator in name_lower for indicator in self.HTTP_INDICATORS) or
                name_lower.startswith(self.HTTP_METHOD_PREFIXES))


@plugin
class HTTPMethodAssertionsPlugin(TraceAssertionsPlugin):
    """Plugin providing HTTP method assertion capabilities."""
    
    @classmethod
    def get_plugin_name(cls) -> str:
        return "http_methods"
    
    def assert_http_method(self, method: str) -> 'TraceAssertions':
        """Assert that spans contain the specified HTTP method."""
        method = method.upper()
        http_spans = [HTTPSpan(span) for span in self._current_spans if HTTPSpan(span).is_http_span]
        matching_spans = [hs for hs in http_spans if hs.method == method]
        
        assert matching_spans, f"No HTTP spans found with method '{method}'. Found methods: {list(set(hs.method for hs in http_spans if hs.method))}"
        
        # Update current spans to only include matching spans
        self._current_spans = [hs.span for hs in matching_spans]
        return self
    
    def assert_http_methods(self, methods: List[str]) -> 'TraceAssertions':
        """Assert that spans contain any of the specified HTTP methods."""
        methods = [m.upper() for m in methods]
        http_spans = [HTTPSpan(span) for span in self._current_spans if HTTPSpan(span).is_http_span]
        matching_spans = [hs for hs in http_spans if hs.method in methods]
        
        assert matching_spans, f"No HTTP spans found with methods {methods}. Found methods: {list(set(hs.method for hs in http_spans if hs.method))}"
        
        self._current_spans = [hs.span for hs in matching_spans]
        return self
    
    # Convenience methods - use assert_http_method() directly for better flexibility
    
    def assert_http_span_type(self, span_type: str) -> 'TraceAssertions':
        """Assert spans of a specific HTTP span type exist.
        
        Args:
            span_type: 'http.process' (server-side) or 'http.send' (client-side)
        """
        valid_types = [HTTPSpan.HTTP_PROCESS_SPAN_TYPE, HTTPSpan.HTTP_SEND_SPAN_TYPE]
        assert span_type in valid_types, f"Invalid span type '{span_type}'. Valid types: {valid_types}"
        
        http_spans = [HTTPSpan(span) for span in self._current_spans if HTTPSpan(span).is_http_span]
        matching_spans = [hs for hs in http_spans if hs.span_type == span_type]
        
        assert matching_spans, f"No spans found with type '{span_type}'. Found span types: {list(set(hs.span_type for hs in http_spans if hs.span_type))}"
        
        self._current_spans = [hs.span for hs in matching_spans]
        return self

    def assert_http_status_code(self, status_code: int) -> 'TraceAssertions':
        """Assert that spans contain the specified HTTP status code."""
        http_spans = [HTTPSpan(span) for span in self._current_spans if HTTPSpan(span).is_http_span]
        matching_spans = [hs for hs in http_spans if hs.status_code == status_code]
        
        assert matching_spans, f"No HTTP spans found with status code {status_code}. Found status codes: {list(set(hs.status_code for hs in http_spans if hs.status_code))}"
        
        self._current_spans = [hs.span for hs in matching_spans]
        return self
    
    def assert_status_code_range(self, min_code: int, max_code: int) -> 'TraceAssertions':
        """Assert that spans contain status codes within the specified range.
        
        Args:
            min_code: Minimum status code (inclusive)
            max_code: Maximum status code (exclusive)
            
        Examples:
            - assert_status_code_range(200, 300)  # Success codes (2xx)
            - assert_status_code_range(400, 500)  # Client errors (4xx) 
            - assert_status_code_range(500, 600)  # Server errors (5xx)
        """
        http_spans = [HTTPSpan(span) for span in self._current_spans if HTTPSpan(span).is_http_span]
        matching_spans = [hs for hs in http_spans 
                         if hs.status_code and min_code <= hs.status_code < max_code]
        
        range_name = f"{min_code}-{max_code-1}"
        assert matching_spans, f"No HTTP spans found with status codes in range {range_name}. Found status codes: {list(set(hs.status_code for hs in http_spans if hs.status_code))}"
        
        self._current_spans = [hs.span for hs in matching_spans]
        return self
    
    def assert_http_route(self, route: str) -> 'TraceAssertions':
        """Assert that spans contain the specified HTTP route."""
        http_spans = [HTTPSpan(span) for span in self._current_spans if HTTPSpan(span).is_http_span]
        matching_spans = [hs for hs in http_spans if hs.route == route]
        
        assert matching_spans, f"No HTTP spans found with route '{route}'. Found routes: {list(set(hs.route for hs in http_spans if hs.route))}"
        
        self._current_spans = [hs.span for hs in matching_spans]
        return self
    
    def assert_http_url_pattern(self, pattern: str) -> 'TraceAssertions':
        """Assert that spans contain URLs matching the specified pattern (regex)."""
        http_spans = [HTTPSpan(span) for span in self._current_spans if HTTPSpan(span).is_http_span]
        regex = re.compile(pattern)
        matching_spans = [hs for hs in http_spans if hs.url and regex.search(hs.url)]
        
        assert matching_spans, f"No HTTP spans found with URL matching pattern '{pattern}'. Found URLs: {list(set(hs.url for hs in http_spans if hs.url))}"
        
        self._current_spans = [hs.span for hs in matching_spans]
        return self


@plugin
class HTTPFlowAssertionsPlugin(TraceAssertionsPlugin):
    """Plugin providing HTTP flow and sequence assertion capabilities."""
    
    @classmethod
    def get_plugin_name(cls) -> str:
        return "http_flow"
    
    def assert_request_response_pairs(self) -> 'TraceAssertions':
        """Assert that HTTP request spans are followed by corresponding response spans."""
        http_spans = [HTTPSpan(span) for span in self._current_spans if HTTPSpan(span).is_http_span]
        ordered_spans = sorted(http_spans, key=lambda hs: hs.span.start_time)
        
        request_spans = [hs for hs in ordered_spans if hs.is_request_span]
        response_spans = [hs for hs in ordered_spans if hs.is_response_span]
        
        assert len(request_spans) > 0, "No HTTP request spans found"
        assert len(response_spans) > 0, "No HTTP response spans found"
        
        # For each request, try to find a corresponding response
        paired_count = 0
        for req_span in request_spans:
            # Find response spans that start after this request
            corresponding_responses = [
                res_span for res_span in response_spans 
                if res_span.span.start_time >= req_span.span.start_time
            ]
            if corresponding_responses:
                paired_count += 1
        
        assert paired_count > 0, f"Found {len(request_spans)} request spans but no corresponding response spans"
        return self
    
    def assert_http_flow_sequence(self, expected_methods: List[str]) -> 'TraceAssertions':
        """Assert that HTTP methods appear in the specified sequence."""
        expected_methods = [m.upper() for m in expected_methods]
        http_spans = [HTTPSpan(span) for span in self._current_spans if HTTPSpan(span).is_http_span]
        ordered_spans = sorted(http_spans, key=lambda hs: hs.span.start_time)
        
        actual_methods = [hs.method for hs in ordered_spans if hs.method]
        
        # Check if the expected sequence appears in the actual sequence
        if len(expected_methods) > len(actual_methods):
            assert False, f"Expected sequence {expected_methods} longer than actual sequence {actual_methods}"
        
        # Find the expected sequence in the actual sequence
        for i in range(len(actual_methods) - len(expected_methods) + 1):
            if actual_methods[i:i+len(expected_methods)] == expected_methods:
                return self  # Found the sequence
        
        assert False, f"Expected HTTP method sequence {expected_methods} not found in actual sequence {actual_methods}"
    
    # Use assert_http_flow_sequence(['POST', 'GET', 'PUT', 'DELETE']) for CRUD flows
    
    def assert_api_call_timing(self, max_duration_ms: float) -> 'TraceAssertions':
        """Assert that HTTP API calls complete within specified time."""
        http_spans = [HTTPSpan(span) for span in self._current_spans if HTTPSpan(span).is_http_span]
        
        for hs in http_spans:
            if hasattr(hs.span, 'end_time') and hasattr(hs.span, 'start_time'):
                duration_ns = hs.span.end_time - hs.span.start_time
                duration_ms = duration_ns / HTTPSpan.NANOSECONDS_TO_MILLISECONDS  # Convert nanoseconds to milliseconds
                
                assert duration_ms <= max_duration_ms, f"HTTP span '{hs.name}' took {duration_ms:.2f}ms, exceeding limit of {max_duration_ms}ms"
        
        return self
    
    def assert_concurrent_requests(self, min_concurrent: int = 2) -> 'TraceAssertions':
        """Assert that there are overlapping (concurrent) HTTP requests."""
        http_spans = [HTTPSpan(span) for span in self._current_spans 
                     if HTTPSpan(span).is_http_span and hasattr(span, 'start_time') and hasattr(span, 'end_time')]
        
        concurrent_count = 0
        for i, span1 in enumerate(http_spans):
            for span2 in http_spans[i+1:]:
                # Check if spans overlap in time
                if (span1.span.start_time < span2.span.end_time and 
                    span2.span.start_time < span1.span.end_time):
                    concurrent_count += 1
                    break  # Found at least one overlap for span1
        
        assert concurrent_count >= min_concurrent, f"Expected at least {min_concurrent} concurrent HTTP requests, found {concurrent_count}"
        return self
    
    def assert_http_method_distribution(self, expected_distribution: Dict[str, int]) -> 'TraceAssertions':
        """Assert the distribution of HTTP methods matches expected counts."""
        http_spans = [HTTPSpan(span) for span in self._current_spans if HTTPSpan(span).is_http_span]
        
        actual_distribution = {}
        for hs in http_spans:
            if hs.method:
                method = hs.method.upper()
                actual_distribution[method] = actual_distribution.get(method, 0) + 1
        
        for method, expected_count in expected_distribution.items():
            method = method.upper()
            actual_count = actual_distribution.get(method, 0)
            assert actual_count == expected_count, f"Expected {expected_count} {method} requests, found {actual_count}"
        
        return self
    
    def assert_rest_api_endpoints(self, endpoints: List[Dict[str, str]]) -> 'TraceAssertions':
        """Assert that specific REST API endpoints were called.
        
        Args:
            endpoints: List of dicts with 'method' and 'path' keys
            Example: [{'method': 'GET', 'path': '/users'}, {'method': 'POST', 'path': '/orders'}]
        """
        http_spans = [HTTPSpan(span) for span in self._current_spans if HTTPSpan(span).is_http_span]
        
        for endpoint in endpoints:
            method = endpoint['method'].upper()
            expected_path = endpoint['path']
            
            # Find spans matching this endpoint
            matching_spans = []
            for hs in http_spans:
                if hs.method == method:
                    # Check URL or route contains the path
                    if ((hs.url and expected_path in hs.url) or 
                        (hs.route and expected_path in hs.route)):
                        matching_spans.append(hs)
            
            assert matching_spans, f"No spans found for endpoint {method} {expected_path}"
        
        return self


@plugin  
class HTTPValidationPlugin(TraceAssertionsPlugin):
    """Plugin providing HTTP validation and debugging utilities."""
    
    @classmethod
    def get_plugin_name(cls) -> str:
        return "http_validation"
    
    def debug_http_spans(self) -> 'TraceAssertions':
        """Print debug information about HTTP spans for troubleshooting."""
        http_spans = [HTTPSpan(span) for span in self._current_spans if HTTPSpan(span).is_http_span]
        
        print(f"\n=== HTTP Spans Debug Info ({len(http_spans)} spans) ===")
        for i, hs in enumerate(http_spans, 1):
            print(f"{i}. Span: {hs.name}")
            print(f"   Span Type: {hs.span_type}")
            print(f"   Method: {hs.method}")
            print(f"   Status: {hs.status_code}")
            print(f"   URL: {hs.url}")
            print(f"   Route: {hs.route}")
            print(f"   Is HTTP Process: {hs.is_http_process_span}")
            print(f"   Is HTTP Send: {hs.is_http_send_span}")
            print(f"   Is Request: {hs.is_request_span}")
            print(f"   Is Response: {hs.is_response_span}")
            print(f"   Attributes: {dict(hs.attributes)}")
            print()
        
        return self
    
    def validate_http_span_completeness(self) -> 'TraceAssertions':
        """Validate that HTTP spans have required attributes."""
        http_spans = [HTTPSpan(span) for span in self._current_spans if HTTPSpan(span).is_http_span]
        
        incomplete_spans = []
        for hs in http_spans:
            issues = []
            if not hs.method:
                issues.append("missing HTTP method")
            if not hs.status_code and not hs.is_request_span:
                issues.append("missing status code for response span")
            if not hs.url and not hs.route:
                issues.append("missing URL or route information")
            
            if issues:
                incomplete_spans.append(f"Span '{hs.name}': {', '.join(issues)}")
        
        assert not incomplete_spans, "Found incomplete HTTP spans:\n" + "\n".join(incomplete_spans)
        return self
    
    def assert_no_http_errors(self) -> 'TraceAssertions':
        """Assert that no HTTP error status codes (4xx, 5xx) are present."""
        http_spans = [HTTPSpan(span) for span in self._current_spans if HTTPSpan(span).is_http_span]
        
        error_spans = [hs for hs in http_spans 
                      if hs.status_code and hs.status_code >= HTTPSpan.CLIENT_ERROR_MIN]
        
        if error_spans:
            error_details = [f"{hs.name}: {hs.method} {hs.status_code}" for hs in error_spans]
            assert False, "Found HTTP error spans:\n" + "\n".join(error_details)
        
        return self