"""
Demo script to verify monocle_trace_method input/output capture functionality.
This demonstrates that the custom span decorator now captures function inputs and outputs.
"""
import json
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tests'))

from monocle_apptrace.instrumentation.common.instrumentor import (
    setup_monocle_telemetry,
    monocle_trace_method
)
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

# Setup telemetry with console exporter to see spans
instrumentor = setup_monocle_telemetry(
    workflow_name="demo_custom_span_io",
    span_processors=[SimpleSpanProcessor(ConsoleSpanExporter())]
)

print("=" * 80)
print("DEMO: monocle_trace_method with Input/Output Capture")
print("=" * 80)

# Test 1: Simple function with args
print("\n1. Testing simple function with positional arguments:")
print("-" * 60)

@monocle_trace_method(span_name="add_numbers")
def add_numbers(x, y):
    """Simple addition function."""
    return x + y

result = add_numbers(5, 10)
print(f"Function call: add_numbers(5, 10)")
print(f"Result: {result}")
print("Expected span events:")
print('  - data.input: {"args": [5, 10], "kwargs": {}}')
print(f'  - data.output: {{"result": {result}}}')

# Test 2: Function with kwargs
print("\n2. Testing function with keyword arguments:")
print("-" * 60)

@monocle_trace_method(span_name="greet_user")
def greet_user(name, greeting="Hello"):
    """Greeting function with default parameter."""
    return f"{greeting}, {name}!"

result = greet_user("Alice", greeting="Hi")
print(f'Function call: greet_user("Alice", greeting="Hi")')
print(f"Result: {result}")
print("Expected span events:")
print('  - data.input: {"args": ["Alice"], "kwargs": {"greeting": "Hi"}}')
print(f'  - data.output: {{"result": "{result}"}}')

# Test 3: Function with complex types
print("\n3. Testing function with complex data types:")
print("-" * 60)

@monocle_trace_method(span_name="process_data")
def process_data(data_dict, items_list):
    """Process dictionary and list."""
    return {
        "processed": True,
        "dict_keys": list(data_dict.keys()),
        "list_count": len(items_list)
    }

input_dict = {"name": "test", "value": 42}
input_list = [1, 2, 3, 4, 5]
result = process_data(input_dict, input_list)
print(f"Function call: process_data({input_dict}, {input_list})")
print(f"Result: {result}")
print("Expected span events:")
print(f'  - data.input: {{"args": [{input_dict}, {input_list}], "kwargs": {{}}}}')
print(f'  - data.output: {{"result": {result}}}')

# Test 4: Function that raises an exception
print("\n4. Testing function that raises an exception:")
print("-" * 60)

@monocle_trace_method(span_name="divide_numbers")
def divide_numbers(x, y):
    """Division function that may raise error."""
    return x / y

try:
    result = divide_numbers(10, 0)
except ZeroDivisionError as e:
    print(f"Function call: divide_numbers(10, 0)")
    print(f"Exception raised: {type(e).__name__}: {e}")
    print("Expected span events:")
    print('  - data.input: {"args": [10, 0], "kwargs": {}}')
    print('  - data.output with error_code: "error"')
    print('  - Span status: ERROR')

print("\n" + "=" * 80)
print("DEMO COMPLETE")
print("=" * 80)
print("\nNote: The actual spans shown above in the console output demonstrate")
print("that the decorator now captures:")
print("  ✓ Function inputs (args and kwargs)")
print("  ✓ Function outputs (return values)")  
print("  ✓ Error information when exceptions occur")
print("  ✓ Span type set to 'custom'")
print("\nThis gives visibility into what was passed to methods and what they returned!")
print("=" * 80)

if instrumentor:
    instrumentor.uninstrument()
