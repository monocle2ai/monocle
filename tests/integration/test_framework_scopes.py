import os
import pytest
from common.custom_exporter import CustomConsoleSpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from monocle_apptrace.instrumentation.common.constants import SCOPE_METHOD_FILE, SCOPE_CONFIG_PATH

custom_exporter = CustomConsoleSpanExporter()

@pytest.fixture(scope="module")
def setup():
    os.environ[SCOPE_CONFIG_PATH] = os.path.join(os.path.dirname(os.path.abspath(__file__)), SCOPE_METHOD_FILE)
    setup_monocle_telemetry(
                workflow_name="framework_scopes_test",
                span_processors=[SimpleSpanProcessor(custom_exporter)],
                wrapper_methods=[])
    
@pytest.fixture(autouse=True)
def pre_test():
    # clear old spans
    custom_exporter.reset()

def framework_scopes_accessor(arguments):
    """Example output_processor accessor that uses frameworks_scopes data"""
    frameworks_scopes = arguments.get("frameworks_scopes", {})
    
    # Access teams.ai framework scopes
    if "teams.ai" in frameworks_scopes:
        return frameworks_scopes["teams.ai"]
    
    return []

@pytest.mark.integration()
def test_load_framework_scopes(setup):
    """Test loading the framework_scopes from configuration file"""
    from monocle_apptrace.instrumentation.common.utils import load_scopes
    
    # Load the scopes configuration
    scope_methods, frameworks = load_scopes()
    
    # Check if frameworks loaded correctly
    assert frameworks is not None
    assert isinstance(frameworks, dict)
    
    # Verify teams.ai framework is present with correct scopes
    assert "teams.ai" in frameworks
    framework_scopes = frameworks["teams.ai"]
    assert len(framework_scopes) == 2
    assert "user" in framework_scopes
    assert "conversation" in framework_scopes

@pytest.mark.integration()
def test_framework_scopes_in_processor(setup):
    """Test accessing framework_scopes in a simulated output processor"""
    # Simulate the arguments dictionary that would be passed to an accessor
    mock_arguments = {
        "instance": None,
        "args": [],
        "kwargs": {},
        "result": None,
        "frameworks_scopes": {"teams.ai": ["user", "conversation"]}
    }
    
    # Call the accessor function that would be used in an output processor
    result = framework_scopes_accessor(mock_arguments)
    
    # Verify results
    assert result is not None
    assert len(result) == 2
    assert "user" in result
    assert "conversation" in result
