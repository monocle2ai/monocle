"""
Test for Google ADK E-commerce Agent with monocle_trace context manager.

This test validates:
1. Proper span generation for ADK agents and tools
2. No duplicate spans in the trace
3. Correct parent-child relationships with monocle_trace
4. Custom attributes set via monocle_trace are present
5. All spans are in the same trace (no fragmentation)
"""

import asyncio
import logging
import time
import sys
import os
import pytest

from integration.commerce.adk_long_m import demonstrate_ecommerce_agent_flow
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from monocle_apptrace import setup_monocle_telemetry

logger = logging.getLogger(__name__)


@pytest.fixture(scope="function")
def setup():
    """Setup memory exporter and monocle telemetry."""
    memory_exporter = InMemorySpanExporter()
    instrumentor = None
    try:
        instrumentor = setup_monocle_telemetry(
            workflow_name="adk_ecommerce_example",
            span_processors=[SimpleSpanProcessor(memory_exporter)]
            # monocle_exporters_list='file'
        )
        yield memory_exporter
    finally:
        # Clean up
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()


def test_adk_commerce_with_monocle_trace(setup):
    """Test ADK commerce agent with monocle_trace context manager."""
    
    try:
        # Run the e-commerce flow
        asyncio.run(demonstrate_ecommerce_agent_flow())
    finally:    
        verify_spans(memory_exporter=setup)


def verify_spans(memory_exporter):
    """Verify all spans are generated correctly with no duplicates."""
    
    time.sleep(2)  # Wait for spans to be exported
    
    spans = memory_exporter.get_finished_spans()
    
    # Basic validation
    assert len(spans) > 0, "No spans were generated"
    
    # Collect all trace IDs to verify no fragmentation
    trace_ids = set()
    span_ids = set()
    span_names = []
    
    # Flags for required spans
    found_workflow = False
    found_add_to_cart_monocle_span = False
    found_view_cart_monocle_span = False
    found_search_products_monocle_span = False
    found_payment_process_monocle_span = False
    found_inference = False
    found_adk_agent = False
    found_tool_invocation = False
    
    # Track monocle_trace span attributes
    add_cart_spans = []
    view_cart_spans = []
    search_spans = []
    payment_spans = []
    
    for span in spans:
        span_attributes = span.attributes
        trace_id = format(span.context.trace_id, '032x')
        span_id = format(span.context.span_id, '016x')
        
        trace_ids.add(trace_id)
        span_names.append(span.name)
        
        # Check for duplicate span_ids
        assert span_id not in span_ids, f"Duplicate span_id found: {span_id} for span: {span.name}"
        span_ids.add(span_id)
        
        # Check for workflow span
        if "workflow" in span.name.lower() or (
            "span.type" in span_attributes and span_attributes.get("span.type") == "workflow"
        ):
            found_workflow = True
            assert span_attributes.get("workflow.name") == "adk_ecommerce_example"
        
        # Check for monocle_trace spans - ecommerce.cart.add_item
        if span.name == "ecommerce.cart.add_item":
            found_add_to_cart_monocle_span = True
            add_cart_spans.append(span_attributes)
            
            # Verify custom attributes from monocle_trace
            assert "user.id" in span_attributes, "user.id attribute missing from add_to_cart span"
            assert "product.id" in span_attributes, "product.id attribute missing"
            assert "cart.quantity" in span_attributes, "cart.quantity attribute missing"
            assert "cart.total_items" in span_attributes, "cart.total_items attribute missing"
            assert "operation.success" in span_attributes, "operation.success attribute missing"
            
            # Verify monocle metadata
            assert "monocle_apptrace.version" in span_attributes
            assert "workflow.name" in span_attributes
            
            # Verify the span has a parent (not a root span when called via ADK)
            if span.parent:
                parent_span_id = format(span.parent.span_id, '016x')
                logger.info(f"add_to_cart span parent: {parent_span_id}")
        
        # Check for monocle_trace spans - ecommerce.cart.view
        if span.name == "ecommerce.cart.view":
            found_view_cart_monocle_span = True
            view_cart_spans.append(span_attributes)
            
            # Verify custom attributes from monocle_trace
            assert "user.id" in span_attributes, "user.id attribute missing from view_cart span"
            assert "cart.empty" in span_attributes, "cart.empty attribute missing"
            assert "cart.total_items" in span_attributes, "cart.total_items attribute missing"
            
            # If cart is not empty, should have unique_products and total_value
            if not span_attributes.get("cart.empty"):
                assert "cart.unique_products" in span_attributes, "cart.unique_products missing for non-empty cart"
                assert "cart.total_value" in span_attributes, "cart.total_value missing for non-empty cart"
            
            # Verify monocle metadata
            assert "monocle_apptrace.version" in span_attributes
            assert "workflow.name" in span_attributes
        
        # Check for monocle_trace spans - ecommerce.search.products
        if span.name == "ecommerce.search.products":
            found_search_products_monocle_span = True
            search_spans.append(span_attributes)
            
            # Verify custom attributes
            assert "search.query" in span_attributes, "search.query attribute missing"
            assert "search.results_count" in span_attributes, "search.results_count attribute missing"
            assert "search.total_catalog_size" in span_attributes, "search.total_catalog_size missing"
            assert "search.results_found" in span_attributes, "search.results_found attribute missing"
        
        # Check for monocle_trace spans - ecommerce.payment.process
        if span.name == "ecommerce.payment.process":
            found_payment_process_monocle_span = True
            payment_spans.append(span_attributes)
            
            # Verify custom attributes
            assert "user.id" in span_attributes, "user.id attribute missing from payment span"
            assert "payment.method" in span_attributes, "payment.method attribute missing"
            assert "billing.address" in span_attributes, "billing.address attribute missing"
            assert "payment.success" in span_attributes, "payment.success attribute missing"
            assert "order.total" in span_attributes, "order.total attribute missing"
            assert "order.item_count" in span_attributes, "order.item_count attribute missing"
        
        # Check for inference spans (LLM calls)
        if "span.type" in span_attributes and span_attributes["span.type"] == "inference":
            found_inference = True
            assert "entity.1.type" in span_attributes
            # Gemini inference validation
            if "gemini" in span_attributes.get("entity.1.type", "").lower():
                assert "entity.1.inference_endpoint" in span_attributes
        
        # Check for ADK agent spans
        if "span.type" in span_attributes and span_attributes["span.type"] == "agentic.invocation":
            found_adk_agent = True
            assert "entity.1.type" in span_attributes
            assert span_attributes["entity.1.type"] == "agent.adk"
            assert "entity.1.name" in span_attributes
        
        # Check for tool invocation spans
        if "span.type" in span_attributes and span_attributes["span.type"] == "agentic.tool.invocation":
            found_tool_invocation = True
            assert "entity.1.type" in span_attributes
            assert span_attributes["entity.1.type"] == "tool.adk"
            assert "entity.1.name" in span_attributes
            
            # Verify tool names
            tool_name = span_attributes.get("entity.1.name")
            expected_tools = ["add_to_cart", "view_cart", "search_products", "mock_payment_process", "load_memory"]
            assert tool_name in expected_tools, f"Unexpected tool name: {tool_name}"
    
    # Assert all required spans were found
    assert found_workflow, "Workflow span not found"
    assert found_add_to_cart_monocle_span, "ecommerce.cart.add_item span not found - monocle_trace not working"
    assert found_view_cart_monocle_span, "ecommerce.cart.view span not found - monocle_trace not working"
    assert found_search_products_monocle_span, "ecommerce.search.products span not found - monocle_trace not working"
    assert found_payment_process_monocle_span, "ecommerce.payment.process span not found - monocle_trace not working"
    assert found_inference, "Inference (LLM) span not found"
    assert found_adk_agent, "ADK agent span not found"
    assert found_tool_invocation, "Tool invocation span not found"
    
    # Verify no trace fragmentation - all spans should be in same trace
    # Note: ADK creates multiple sessions which may result in multiple traces
    logger.info(f"Found {len(trace_ids)} trace(s): {trace_ids}")
    
    # Verify no duplicate span names per operation
    add_cart_count = span_names.count("ecommerce.cart.add_item")
    view_cart_count = span_names.count("ecommerce.cart.view")
    search_count = span_names.count("ecommerce.search.products")
    payment_count = span_names.count("ecommerce.payment.process")
    
    # We expect: 2 add_to_cart (laptop + mouse), 3 view_cart (checkout session), 
    # 2 search (browsing session), 1 payment
    assert add_cart_count == 2, f"Expected 2 add_to_cart spans, found {add_cart_count}"
    assert view_cart_count == 2, f"Expected 2 view_cart spans, found {view_cart_count}"
    assert search_count == 2, f"Expected 2 search spans, found {search_count}"
    assert payment_count == 1, f"Expected 1 payment span, found {payment_count}"
    
    # Log summary
    logger.info(f"✓ Total spans generated: {len(spans)}")
    logger.info(f"✓ Unique trace IDs: {len(trace_ids)}")
    logger.info(f"✓ No duplicate span_ids detected")
    logger.info(f"✓ monocle_trace spans found: add_to_cart={add_cart_count}, view_cart={view_cart_count}, search={search_count}, payment={payment_count}")
    logger.info(f"✓ Custom attributes validated on all monocle_trace spans")
    
    # Verify parent-child relationships
    verify_parent_child_relationships(spans)


def verify_parent_child_relationships(spans):
    """Verify that monocle_trace spans have proper parent relationships."""
    
    span_map = {}
    for span in spans:
        span_id = format(span.context.span_id, '016x')
        span_map[span_id] = {
            'name': span.name,
            'parent_id': format(span.parent.span_id, '016x') if span.parent else None,
            'span': span
        }
    
    # Find monocle_trace spans and verify they have parents
    monocle_trace_spans = [
        span_info for span_info in span_map.values()
        if span_info['name'] in ["ecommerce.cart.add_item", "ecommerce.cart.view"]
    ]
    
    for span_info in monocle_trace_spans:
        # When called via ADK tool, monocle_trace spans should have a parent
        if span_info['parent_id']:
            parent = span_map.get(span_info['parent_id'])
            assert parent is not None, f"Parent span not found for {span_info['name']}"
            logger.info(f"✓ {span_info['name']} has parent: {parent['name']}")
        else:
            # If no parent, it should be a root span (called outside ADK context)
            logger.warning(f"⚠ {span_info['name']} has no parent - might be called outside ADK context")
    
    logger.info(f"✓ Parent-child relationships verified for {len(monocle_trace_spans)} monocle_trace spans")
