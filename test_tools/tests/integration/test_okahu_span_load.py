from asyncio import sleep
import pytest
import os

from requests import HTTPError
import requests

from monocle_apptrace.exporters.okahu.okahu_eval_result_exporter import OkahuEvalResultExporter
from monocle_test_tools import TraceAssertion
from monocle_apptrace.exporters.okahu.okahu_exporter import OkahuSpanExporter
from test_common.okahu_api import OkahuAPI
TRACE_ID="ac569fb7e6fc1fc3fa68bff4b468a4f7"
WORKFLOW_NAME="test_fs_financial_agent"
TEST_TRACE_FILE = os.path.join(os.path.dirname(__file__), f"test_data/monocle_{TRACE_ID}_trace.json")

#add a fixture to export the trace data 
@pytest.fixture(autouse=True)
def export_trace_to_okahu(monocle_trace_asserter: TraceAssertion):
    # Create workflow
    try:
        OkahuAPI().create_workflow(name=WORKFLOW_NAME, description="A test workflow created via API")
    except HTTPError as e:
        if e.response.status_code == 409:
            # Workflow already exists, ignore
            pass
        else:
            raise
    yield monocle_trace_asserter

@pytest.mark.asyncio
# skip if OKAHU_API_KEY is not set
@pytest.mark.skipif(not os.getenv("OKAHU_API_KEY"), reason="OKAHU_API_KEY is not set")
async def test_spans_load_from_okahu(monocle_trace_asserter:TraceAssertion):
    try:
        monocle_trace_asserter.with_trace_source("okahu", id=TRACE_ID, workflow_name=WORKFLOW_NAME)
    except HTTPError as e:
        if e.response.status_code == 404:
            # If the trace is not found, export it first
            monocle_trace_asserter.with_trace_source("file", trace_path=TEST_TRACE_FILE)
            exporter = OkahuSpanExporter()
            exporter.export(monocle_trace_asserter.validator.spans)
            exporter.force_flush()
            # Retry loading the trace from Okahu after exporting
            monocle_trace_asserter.with_trace_source("okahu", id=TRACE_ID, workflow_name=WORKFLOW_NAME)
        else:
            raise

    monocle_trace_asserter.called_agent("okahu_demo_fs_agent_supervisor")
    monocle_trace_asserter.called_tool("okahu_demo_fs_tool_get_portfolio")

    def _create_okahu_workflow(self, name: str, description: str) -> dict:
        """
        Create a new workflow in the Okahu API.

        Args:
            name (str): The name of the workflow.
            description (str): The description of the workflow.

        Returns:
            dict: The created workflow details.
        """
        url = f"{self.api_url}/v1/components/{name.replace(' ', '_').lower()}"
        headers = self.headers
        payload = {
            "display_name": name,
            "description": description,
            "type": "workflow.generic",
            "domain": "logical",
            "status": "active"
        }

        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()

if __name__ == "__main__":
    pytest.main([__file__])