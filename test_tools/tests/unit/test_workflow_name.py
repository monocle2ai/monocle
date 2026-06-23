import pytest
import os
from opentelemetry.context import get_value
from monocle_apptrace.instrumentation.common.constants import MONOCLE_WORKFLOW_NAME_KEY
from monocle_test_tools import MonocleValidator
TEST_WORKFLOW_NAME = "test_workflow"

@pytest.fixture(scope="module")
def setup():
    os.environ["MONOCLE_TEST_WORKFLOW_NAME"] = TEST_WORKFLOW_NAME
    # Reset singleton so MonocleValidator() re-initializes and picks up the env var
    MonocleValidator._initialized = False
    MonocleValidator._instance = None

def test_workflow_env(setup):
    #verify that the workflow name is set correctly in the validator
    validator = MonocleValidator()
    assert get_value(MONOCLE_WORKFLOW_NAME_KEY) == TEST_WORKFLOW_NAME