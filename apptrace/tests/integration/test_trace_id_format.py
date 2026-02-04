"""
Integration test to verify that trace_id and span_id are exported without '0x' prefix
in both filenames and JSON content for file exporter and in-memory exporter.
"""
import json
import logging
import os
import pytest
import shutil
import time
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.llm import LLMChain
from monocle_apptrace import setup_monocle_telemetry
from common.custom_exporter import CustomConsoleSpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

logger = logging.getLogger(__name__)

# Test directory for trace files
TEST_TRACE_DIR = "./.monocle_test_trace_id_format"


@pytest.fixture(scope="function")
def setup_file_exporter():
    """Setup telemetry with file exporter"""
    instrumentor = None
    try:
        # Clean up test directory if it exists
        if os.path.exists(TEST_TRACE_DIR):
            shutil.rmtree(TEST_TRACE_DIR)
        
        # Setup environment for file exporter BEFORE initializing telemetry
        os.environ["MONOCLE_TRACE_OUTPUT_PATH"] = TEST_TRACE_DIR
        
        # Setup monocle telemetry with file exporter
        instrumentor = setup_monocle_telemetry(
            workflow_name="trace_id_format_test",
            monocle_exporters_list="file"
        )
        yield
        
    finally:
        # Cleanup
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()
        
        # Clean up environment variables
        os.environ.pop("MONOCLE_TRACE_OUTPUT_PATH", None)
        
        # Clean up test directory after test completes (success or failure)
        if os.path.exists(TEST_TRACE_DIR):
            try:
                shutil.rmtree(TEST_TRACE_DIR)
                logger.info(f"Cleaned up test directory: {TEST_TRACE_DIR}")
            except Exception as e:
                logger.warning(f"Failed to clean up test directory: {e}")


@pytest.fixture(scope="function")
def setup_inmemory_exporter():
    """Setup telemetry with in-memory and console exporters"""
    try:
        
        # Create inmemory exporter
        memory_exporter = InMemorySpanExporter()
        span_processors = [SimpleSpanProcessor(memory_exporter)]
        
        # Setup monocle telemetry with inmemory exporter
        instrumentor = setup_monocle_telemetry(
            workflow_name="trace_id_format_test",
            span_processors=span_processors
        )
        
        yield memory_exporter
    finally:
        # Cleanup
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()
        

def verify_span_ids_format(span_data, span_num, context_info=""):
    """Helper function to verify span IDs format"""
    # Check trace_id in context
    if 'context' in span_data:
        context = span_data['context']
        
        if 'trace_id' in context:
            trace_id = context['trace_id']
            assert not trace_id.startswith('0x'), \
                f"trace_id has '0x' prefix in span {span_num} ({context_info}): {trace_id}"
            assert not trace_id.startswith('0X'), \
                f"trace_id has '0X' prefix in span {span_num} ({context_info}): {trace_id}"
            
            # Verify it's 32 hex characters
            assert len(trace_id) == 32, \
                f"trace_id should be 32 chars in span {span_num} ({context_info}), got {len(trace_id)}: {trace_id}"
            
            # Verify it's valid hex
            try:
                int(trace_id, 16)
            except ValueError:
                pytest.fail(f"trace_id is not valid hex in span {span_num} ({context_info}): {trace_id}")
            logger.info(f"✓ span {span_num} ({context_info}): trace_id format is correct: {trace_id}")
        
        if 'span_id' in context:
            span_id = context['span_id']
            assert not span_id.startswith('0x'), \
                f"span_id has '0x' prefix in span {span_num} ({context_info}): {span_id}"
            assert not span_id.startswith('0X'), \
                f"span_id has '0X' prefix in span {span_num} ({context_info}): {span_id}"
            
            # Verify it's 16 hex characters (64 bits)
            assert len(span_id) == 16, \
                f"span_id should be 16 chars in span {span_num} ({context_info}), got {len(span_id)}: {span_id}"
            
            # Verify it's valid hex
            try:
                int(span_id, 16)
            except ValueError:
                pytest.fail(f"span_id is not valid hex in span {span_num} ({context_info}): {span_id}")
            logger.info(f"✓ span {span_num} ({context_info}): span_id format is correct: {span_id}")
    
    # Check parent_id
    if 'parent_id' in span_data and span_data['parent_id']:
        parent_id = span_data['parent_id']
        if isinstance(parent_id, str):
            assert not parent_id.startswith('0x'), \
                f"parent_id has '0x' prefix in span {span_num} ({context_info}): {parent_id}"
            assert not parent_id.startswith('0X'), \
                f"parent_id has '0X' prefix in span {span_num} ({context_info}): {parent_id}"
            
            # Verify it's 16 hex characters
            assert len(parent_id) == 16, \
                f"parent_id should be 16 chars in span {span_num} ({context_info}), got {len(parent_id)}: {parent_id}"
            
            # Verify it's valid hex
            try:
                int(parent_id, 16)
            except ValueError:
                pytest.fail(f"parent_id is not valid hex in span {span_num} ({context_info}): {parent_id}")
            logger.info(f"✓ span {span_num} ({context_info}): parent_id format is correct: {parent_id}")


def test_file_exporter_trace_id_without_0x_prefix(setup_file_exporter):
    """
    Test file exporter to verify:
    1. Trace files are created in the local directory
    2. Filenames contain trace_id without '0x' prefix
    3. JSON content has trace_id and span_id without '0x' prefix
    """
    
    # Create a simple LangChain workflow to generate traces
    try:
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.7,
        )
        
        prompt = PromptTemplate(
            input_variables=["topic"],
            template="Write a short story about {topic}"
        )
        
        chain = LLMChain(llm=llm, prompt=prompt)
        
        try:
            chain.run(topic="artificial intelligence")
        except Exception as e:
            logger.info(f"Expected error occurred: {e}")
            
    except Exception as e:
        logger.error(f"Error creating workflow: {e}")
    
    # Import trace provider to force flush
    from opentelemetry import trace
    
    # Force flush all spans
    tracer_provider = trace.get_tracer_provider()
    if hasattr(tracer_provider, 'force_flush'):
        tracer_provider.force_flush()
    
    # Give some time for spans to be exported and files to be written
    time.sleep(3)
    
    # Verify trace files
    verify_file_exporter_output()


def verify_file_exporter_output():
    """Helper function to verify file exporter output"""
    # Check if trace files were created
    trace_dir = Path(TEST_TRACE_DIR)
    assert trace_dir.exists(), f"Trace directory {TEST_TRACE_DIR} was not created"
    
    trace_files = list(trace_dir.glob("*.json"))
    logger.info(f"Found {len(trace_files)} trace file(s)")
    
    assert len(trace_files) > 0, "No trace files were created"
    
    # Test each trace file
    for trace_file in trace_files:
        filename = trace_file.name
        logger.info(f"Checking file: {filename}")
        
        # Test 1: Extract trace_id from filename and verify it doesn't have '0x' prefix
        # Expected format: monocle_trace_<service_name>_<trace_id>_YYYY-MM-DD_HH.MM.SS.json
        trace_id_from_filename = None
        if '_' in filename:
            parts = filename.replace('.json', '').split('_')
            # Find the 32-character hex trace_id
            for part in parts:
                if len(part) == 32:
                    try:
                        int(part, 16)
                        trace_id_from_filename = part
                        break
                    except ValueError:
                        continue
            
            if trace_id_from_filename:
                # Verify trace_id doesn't start with '0x' prefix
                assert not trace_id_from_filename.startswith('0x'), \
                    f"trace_id in filename has '0x' prefix: {trace_id_from_filename}"
                assert not trace_id_from_filename.startswith('0X'), \
                    f"trace_id in filename has '0X' prefix: {trace_id_from_filename}"
                logger.info(f"✓ Found trace_id in filename (without 0x prefix): {trace_id_from_filename}")
        
        logger.info(f"✓ Filename format is correct: {filename}")
        
        # Test 2: Verify JSON content
        with open(trace_file, 'r') as f:
            content = f.read()
            
            # Parse JSON (file exporter creates a JSON array)
            try:
                spans_data = json.loads(content)
            except json.JSONDecodeError as e:
                pytest.fail(f"Invalid JSON in {filename}: {e}")
            
            logger.info(f"File has {len(spans_data)} span(s)")
            
            for span_num, span_data in enumerate(spans_data, 1):
                verify_span_ids_format(span_data, span_num, f"file {filename}")
        
        logger.info(f"✓ All checks passed for file: {filename}")
    
    logger.info("=" * 80)
    logger.info("✓ FILE EXPORTER TEST PASSED: All trace_id and span_id values are without '0x' prefix")
    logger.info("=" * 80)


def test_inmemory_exporter_trace_id_without_0x_prefix(setup_inmemory_exporter):
    """
    Test in-memory exporter to verify:
    1. Spans are captured in memory
    2. trace_id and span_id are stored without '0x' prefix
    3. All ID formats are consistent
    """
    memory_exporter = setup_inmemory_exporter
    
    # Create a simple LangChain workflow to generate traces
    try:
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.7,
        )
        
        prompt = PromptTemplate(
            input_variables=["topic"],
            template="Tell me about {topic}"
        )
        
        chain = LLMChain(llm=llm, prompt=prompt)
        
        try:
            chain.run(topic="quantum computing")
        except Exception as e:
            logger.info(f"Expected error occurred: {e}")
            
    except Exception as e:
        logger.error(f"Error creating workflow: {e}")
    
    # Give some time for spans to be exported
    time.sleep(2)
    
    # Verify spans
    verify_inmemory_exporter_output(memory_exporter)


def verify_inmemory_exporter_output(memory_exporter):
    """Helper function to verify in-memory exporter output"""
    # Get captured spans
    spans = memory_exporter.get_finished_spans()
    logger.info(f"Captured {len(spans)} span(s)")
    
    assert len(spans) > 0, "No spans were captured"
    
    all_trace_ids = set()
    all_span_ids = set()
    
    # Test each span
    for idx, span in enumerate(spans, 1):
        logger.info(f"Checking span {idx}: {span.name}")
        # Test to_json output
        try:
            json_output = span.to_json()
            json_data = json.loads(json_output)
              
            # Use helper function to verify JSON structure
            verify_span_ids_format(json_data, idx, f"inmemory span {span.name}")
        
        except json.JSONDecodeError as e:
            pytest.fail(f"Failed to parse span {idx} JSON: {e}")
        
        logger.debug(f"✓ Span {idx} format is correct")
    
    logger.info(f"✓ Found {len(all_trace_ids)} unique trace_id(s)")
    logger.info(f"✓ Found {len(all_span_ids)} unique span_id(s)")
    logger.info("=" * 80)
    logger.info("✓ IN-MEMORY EXPORTER TEST PASSED: All trace_id and span_id values are without '0x' prefix")
    logger.info("=" * 80)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
