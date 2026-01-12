import unittest
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry, get_monocle_instrumentor

class TestHandler(unittest.TestCase):

    instrumentor = None

    def setUp(self):
        """Set up test environment with clean state"""
        # Clean up any existing instrumentor state
        existing_instrumentor = get_monocle_instrumentor()
        if existing_instrumentor is not None:
            try:
                existing_instrumentor.uninstrument()
            except:
                pass

    def tearDown(self) -> None:
        """Clean up instrumentation state"""
        try:
            if self.instrumentor is not None:
                self.instrumentor.uninstrument()
        except Exception as e:
            print("Uninstrument failed:", e)
                
        return super().tearDown()

    def test_call_multi_setup_telemetry(self):
        """
        Test that multiple calls to setup_monocle_telemetry return the same instrumentor instance.
        """
        workflow_name = "test_workflow"

        # First setup
        instrumentor1 = setup_monocle_telemetry(workflow_name)

        # Second setup
        instrumentor2 = setup_monocle_telemetry(workflow_name)
        
        assert instrumentor1 is instrumentor2, "Multiple calls to setup_monocle_telemetry should return the same instrumentor instance."
        
        # Store for cleanup
        self.instrumentor = instrumentor1


if __name__ == '__main__':
    unittest.main()
