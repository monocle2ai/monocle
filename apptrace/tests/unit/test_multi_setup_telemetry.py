from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
def test_call_multi_setup_telemetry():
    """
    Test that multiple calls to setup_monocle_telemetry return the same instrumentor instance.
    """
    workflow_name = "test_workflow"

    # First setup
    instrumentor1 = setup_monocle_telemetry(workflow_name)

    # Second setup
    instrumentor2 = setup_monocle_telemetry(workflow_name)
    
    assert instrumentor1 is instrumentor2, "Multiple calls to setup_monocle_telemetry should return the same instrumentor instance."
