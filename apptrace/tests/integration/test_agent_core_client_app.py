import os
import json
import logging
import boto3
import uuid
from typing import Dict, Any, Optional
import pytest
from common.custom_exporter import CustomConsoleSpanExporter
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from monocle_apptrace.instrumentation.metamodel.botocore.handlers.botocore_span_handler import (
    BotoCoreSpanHandler,
)
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
logger = logging.getLogger(__name__)

@pytest.fixture(scope="module")
def setup():
    try:
        custom_exporter = CustomConsoleSpanExporter()
        instrumentor = setup_monocle_telemetry(
            workflow_name="bedrock_integration_tests",
            span_processors=[SimpleSpanProcessor(custom_exporter)],
            span_handlers={"botocore_handler": BotoCoreSpanHandler()},
        )
        yield custom_exporter
    finally:
        # Clean up instrumentor to avoid global state leakage
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()

class AgentCoreClient:

    def __init__(self, runtime_url: Optional[str] = None, session_id: Optional[str] = None, region: Optional[str] = None):
        """
        Initialize the AgentCore client
        Args:
            runtime_url: The runtime ARN for the AgentCore app (defaults to env var AGENTCORE_RUNTIME_URL)
            session_id: Optional session ID for maintaining conversation context (must be at least 33 chars)
            region: AWS region (defaults to env var AWS_REGION or us-east-1)
        """
        self.runtime_url = runtime_url or os.getenv("AGENTCORE_RUNTIME_URL")
        # Generate a valid session ID (min 33 chars) if not provided
        self.session_id = session_id or f"session-{str(uuid.uuid4())}"
        self.region = region or os.getenv("AWS_REGION", "us-east-1")

        if not self.runtime_url:
            raise ValueError(
                "AGENTCORE_RUNTIME_URL must be set in environment or passed as parameter."
            )

        # Parse the ARN to extract runtime ARN and qualifier
        self.runtime_arn, self.qualifier = self._parse_runtime_arn()

        # Initialize boto3 client for bedrock-agentcore
        self.client = boto3.client('bedrock-agentcore', region_name=self.region)

    def _parse_runtime_arn(self):
        """
        Parse the runtime URL to extract runtime ARN and qualifier
        ARN format: arn:aws:bedrock-agentcore:region:account:runtime/agent-id/runtime-endpoint/DEFAULT
        Returns: (runtime_arn, qualifier)
        """
        # Split by '/runtime-endpoint/'
        if '/runtime-endpoint/' in self.runtime_url:
            parts = self.runtime_url.split('/runtime-endpoint/')
            runtime_arn = parts[0]  # Everything before /runtime-endpoint/
            qualifier = parts[1] if len(parts) > 1 else 'DEFAULT'  # Everything after
            return runtime_arn, qualifier
        else:
            return self.runtime_url, 'DEFAULT'

    def invoke(self, prompt: str, session_id: Optional[str] = None) -> str:
        """
        Args:
            prompt: The user's travel booking request
            session_id: Optional session ID to override the default
        Returns:
            The agent's response text
        """
        sid = session_id or self.session_id

        payload = {
            "prompt": prompt
        }

        try:
            response = self.client.invoke_agent_runtime(
                agentRuntimeArn=self.runtime_arn,
                qualifier=self.qualifier,
                runtimeSessionId=sid,
                payload=json.dumps(payload).encode('utf-8')
            )

            # Parse the response
            if 'response' in response:
                return response['response']._raw_stream.data

            # If no payload, return full response for debugging
            return f"Unexpected response format: {json.dumps(response, default=str, indent=2)}"

        except Exception as e:
            import traceback
            return f"Error invoking AgentCore app: {str(e)}\n{traceback.format_exc()}"

    def _extract_agent_id(self) -> str:
        """Extract agent ID from the runtime ARN"""
        # ARN format: arn:aws:bedrock-agentcore:region:account:runtime/agent-id/runtime-endpoint/DEFAULT
        parts = self.runtime_url.split('/')
        if len(parts) >= 2:
            return parts[-3]  # Get agent-id part
        raise ValueError(f"Invalid runtime ARN format: {self.runtime_url}")

    def _extract_alias_id(self) -> str:
        """Extract alias ID from the runtime ARN"""
        # For now, using DEFAULT or extracting from ARN
        parts = self.runtime_url.split('/')
        if len(parts) >= 4:
            return parts[-1]  # Get DEFAULT or alias name
        return "DEFAULT"

    def invoke_with_context(self, prompt: str, context: Dict[str, Any], session_id: Optional[str] = None) -> str:
        """
        Invoke the AgentCore travel agent with additional context

        Args:
            prompt: The user's travel booking request
            context: Additional context to pass to the agent
            session_id: Optional session ID to override the default

        Returns:
            The agent's response text
        """
        sid = session_id or self.session_id

        payload = {
            "prompt": prompt,
            "context": context
        }

        try:
            response = self.client.invoke_agent_runtime(
                agentRuntimeArn=self.runtime_arn,
                qualifier=self.qualifier,
                runtimeSessionId=sid,
                payload=json.dumps(payload).encode('utf-8')
            )

            # Parse the response
            if 'payload' in response:
                payload_stream = response['payload']
                # Read the streaming response
                if hasattr(payload_stream, 'read'):
                    result_bytes = payload_stream.read()
                    result = result_bytes.decode('utf-8') if isinstance(result_bytes, bytes) else str(result_bytes)
                    # Try to parse as JSON first
                    try:
                        result_json = json.loads(result)
                        return result_json if isinstance(result_json, str) else json.dumps(result_json, indent=2)
                    except json.JSONDecodeError:
                        return result
                else:
                    return str(payload_stream)

            # If no payload, return full response for debugging
            return f"Unexpected response format: {json.dumps(response, default=str, indent=2)}"

        except Exception as e:
            import traceback
            return f"Error invoking AgentCore app: {str(e)}\n{traceback.format_exc()}"

def check_span(spans, exception):
    for span in spans:
        span_attributes = span.attributes

        if "span.type" in span_attributes and (
            span_attributes["span.type"] == "inference" or span_attributes["span.type"] == "inference.framework"):
            # Assertions for all inference attributes

            assert span_attributes["entity.1.type"] == "inference.aws_bedrock"
            assert "entity.1.provider_name" in span_attributes
            assert "entity.1.inference_endpoint" in span_attributes
            assert span_attributes["entity.2.name"] == "okahu_demo_Agent-A8DqSTHJpU"
            assert span_attributes["entity.2.type"] == "model.llm.okahu_demo_Agent-A8DqSTHJpU"
            assert not span.name.lower().startswith("openai")

            # Assertions for metadata
            span_input, span_output, span_metadata = span.events
            assert "input" in span_input.attributes
            assert span_input.attributes["input"] is not None and span_input.attributes["input"] != ""
            assert "response" in span_output.attributes
            assert span_output.attributes["response"] is not None and span_output.attributes["response"] != ""

        if not span.parent and span.name == "workflow":  # Root span
            assert span_attributes["entity.1.name"] == "boto_coverage_api"
            assert span_attributes["entity.1.type"] == "workflow.generic"

def test_agent_core_client(setup):
    """Example usage of the AgentCore client"""
    client = AgentCoreClient()

    print("AWS AgentCore Travel Agent Client")
    print("=" * 50)
    print(f"Connected to: {client.runtime_url}")
    print(f"Session ID: {client.session_id}")
    print("=" * 50)

    user_input = "Book a flight from San Jose to Seattle for 30 March 2026"

    print("\nProcessing your request...")
    response = client.invoke(user_input)
    print(f"\nAgent: {response}")
    spans = setup.get_captured_spans()


if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])

# {
#     "name": "botocore.client.BedrockAgentcore",
#     "context": {
#         "trace_id": "0x00141680b9a355f0e8610c06a0a2fc56",
#         "span_id": "0x0139927e30ead141",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x9f490179aa9c4af5",
#     "start_time": "2026-02-04T15:10:09.438521Z",
#     "end_time": "2026-02-04T15:10:22.887486Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.7.1",
#         "monocle_apptrace.language": "python",
#         "span_source": "",
#         "workflow.name": "bedrock_integration_tests",
#         "entity.1.type": "inference.aws_bedrock",
#         "entity.1.inference_endpoint": "https://bedrock-agentcore.us-east-1.amazonaws.com",
#         "entity.1.provider_name": "bedrock-agentcore.us-east-1.amazonaws.com",
#         "entity.2.name": "okahu_demo_Agent-A8DqSTHJpU",
#         "entity.2.type": "model.llm.okahu_demo_Agent-A8DqSTHJpU",
#         "span.type": "inference",
#         "entity.count": 2,
#         "span.subtype": "turn_end"
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2026-02-04T15:10:09.439150Z",
#             "attributes": {
#                 "input": [
#                     "b'{\"prompt\": \"Book a flight from San Jose to Seattle for 30 March 2026\"}'"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2026-02-04T15:10:22.887486Z",
#             "attributes": {
#                 "response": "b'\"Great! I\\'ve successfully booked your flight from San Jose to Seattle for March 30, 2026. Your booking is confirmed!\"'"
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2026-02-04T15:10:22.887486Z",
#             "attributes": {
#                 "finish_reason": "end_turn",
#                 "finish_type": "success"
#             }
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "bedrock_integration_tests"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "workflow",
#     "context": {
#         "trace_id": "0x00141680b9a355f0e8610c06a0a2fc56",
#         "span_id": "0x9f490179aa9c4af5",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2026-02-04T15:10:09.438521Z",
#     "end_time": "2026-02-04T15:10:22.903210Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.7.1",
#         "monocle_apptrace.language": "python",
#         "span_source": "",
#         "workflow.name": "bedrock_integration_tests",
#         "span.type": "workflow",
#         "entity.1.name": "bedrock_integration_tests",
#         "entity.1.type": "workflow.generic",
#         "entity.2.type": "app_hosting.aws_agentcore",
#         "entity.2.name": "arn:aws:bedrock-agentcore:us-east-1:390041016107:runtime/okahu_demo_Agent-A8DqSTHJpU/runtime-endpoint/DEFAULT",
#         "last.inference": "0x139927e30ead141:*"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "bedrock_integration_tests"
#         },
#         "schema_url": ""
#     }
# }
