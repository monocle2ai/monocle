"""
End-to-End Example: Generate and Validate Traces with Custom Scopes

This example demonstrates:
1. Generating traces tagged with custom scopes in your application
2. Validating those traces using the enhanced import_traces() functionality

Run this in two steps:
    Step 1: Generate traces
    python examples/custom_scope_e2e_example.py --generate

    Step 2: Validate traces
    python examples/custom_scope_e2e_example.py --validate
"""

import os
import sys
import argparse
from datetime import datetime
from typing import Optional, Callable

# Application-side imports (for trace generation)
from monocle_apptrace.utils import start_scope, stop_scope
from monocle_apptrace.instrumentation.common.wrapper_method import monocle_trace_method

# Test-side imports (for trace validation)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../test_tools/src'))
from monocle_test_tools.fluent_api import TraceAssertion


# ============================================================================
# APPLICATION CODE: Generate Traces with Custom Scopes
# ============================================================================

@monocle_trace_method
def search_knowledge_base(query: str) -> str:
    """Simulate a search tool"""
    return f"Search results for: {query}"


@monocle_trace_method
def call_llm(prompt: str) -> str:
    """Simulate an LLM call"""
    return f"LLM response to: {prompt}"


def run_agent_with_test_scope(test_id: str, user_query: str):
    """
    Example: Agent that uses a custom test_id scope to tag traces.
    
    This allows grouping all traces from a single test run.
    """
    scope_token = start_scope("test_id", test_id)
    
    try:
        search_results = search_knowledge_base(user_query)
        prompt = f"Based on: {search_results}, answer: {user_query}"
        response = call_llm(prompt)
        followup = search_knowledge_base("related information")
        return response
    finally:
        stop_scope(scope_token)


def run_agent_with_user_scope(user_id: str, user_query: str):
    """
    Example: Agent that uses a custom user_id scope to tag traces.
    
    This allows tracking all operations for a specific user.
    """
    scope_token = start_scope("user_id", user_id)
    
    try:
        results = search_knowledge_base(f"user {user_id} profile")
        response = call_llm(user_query)
        return response
    finally:
        stop_scope(scope_token)


def run_ci_pipeline_with_scope(ci_run_id: str):
    """
    Example: CI/CD pipeline that tags all traces with a run ID.
    
    This allows validating entire CI runs.
    """
    scope_token = start_scope("ci_run_id", ci_run_id)
    
    try:
        for i in range(3):
            query = f"test query {i+1}"
            result = search_knowledge_base(query)
            response = call_llm(query)
    finally:
        stop_scope(scope_token)


# ============================================================================
# TEST CODE: Validate Traces Using Custom Scopes
# ============================================================================

def validate_scope(scope_name: str, scope_id: str, workflow_name: str, 
                   additional_assertions: Optional[Callable] = None):
    """
    Generic validation function for traces with custom scopes.
    
    Args:
        scope_name: The scope name (e.g., "test_id", "user_id")
        scope_id: The scope value (e.g., "test_123", "user_456")
        workflow_name: Okahu workflow name
        additional_assertions: Optional function that takes asserter and runs more assertions
    """
    asserter = TraceAssertion()
    
    asserter.import_traces(
        trace_source="okahu",
        id=scope_id,
        fact_name="scope",
        scope_name=scope_name,
        workflow_name=workflow_name
    )
    
    # Run additional assertions if provided
    if additional_assertions:
        additional_assertions(asserter)
    
    return asserter


def validate_test_scope(test_id: str, workflow_name: str):
    """
    Validate traces generated with a test_id scope.
    """
    def assertions(asserter):
        asserter.called_tool("search_knowledge_base")
        asserter.called_tool("call_llm")
        asserter.called_tool("search_knowledge_base").has_input("user_query")
        asserter.under_token_limit(10000)
        asserter.under_duration(30, units="seconds")
    
    return validate_scope("test_id", test_id, workflow_name, assertions)


def validate_user_scope(user_id: str, workflow_name: str):
    """
    Validate traces generated with a user_id scope.
    """
    def assertions(asserter):
        asserter.called_tool("search_knowledge_base").has_input(user_id)
        asserter.called_tool("call_llm")
    
    return validate_scope("user_id", user_id, workflow_name, assertions)


def validate_ci_scope(ci_run_id: str, workflow_name: str):
    """
    Validate traces generated with a ci_run_id scope.
    """
    def assertions(asserter):
        asserter.called_tool("search_knowledge_base")
        asserter.called_tool("call_llm")
        asserter.under_token_limit(50000)
        asserter.under_duration(120, units="seconds")
    
    return validate_scope("ci_run_id", ci_run_id, workflow_name, assertions)


# ============================================================================
# MAIN: Command Line Interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="End-to-End Custom Scope Example"
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate traces with custom scopes"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate traces using custom scopes"
    )
    parser.add_argument(
        "--example",
        choices=["test", "user", "ci"],
        default="test",
        help="Which example to run (default: test)"
    )
    parser.add_argument(
        "--workflow",
        default="my_workflow",
        help="Workflow name in Okahu (default: my_workflow)"
    )
    
    args = parser.parse_args()
    
    if not args.generate and not args.validate:
        parser.error("Must specify --generate or --validate")
    
    # Generate unique IDs for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.example == "test":
        test_id = f"test_{timestamp}"
        
        if args.generate:
            run_agent_with_test_scope(test_id, "What is machine learning?")
        
        if args.validate:
            validate_test_scope(test_id, args.workflow)
    
    elif args.example == "user":
        user_id = f"user_{timestamp}"
        
        if args.generate:
            run_agent_with_user_scope(user_id, "Show my recommendations")
        
        if args.validate:
            validate_user_scope(user_id, args.workflow)
    
    elif args.example == "ci":
        ci_run_id = f"ci_run_{timestamp}"
        
        if args.generate:
            run_ci_pipeline_with_scope(ci_run_id)
        
        if args.validate:
            validate_ci_scope(ci_run_id, args.workflow)


if __name__ == "__main__":
    main()
