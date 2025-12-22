import pytest
import os
from opentelemetry.context import get_value
from monocle_test_tools import MonocleValidator
TEST_WORKFLOW_NAME = "test_workflow"

@pytest.fixture(scope="module")
def setup():
    os.environ["MONOCLE_TEST_WORKFLOW_NAME"] = TEST_WORKFLOW_NAME

def test_workflow_env(setup):
    #verify that the workflow name is set correctly in the validator
    validator = MonocleValidator()
    assert get_value("workflow_name") == TEST_WORKFLOW_NAME