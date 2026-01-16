import asyncio
import logging
import os
import json
import pytest
import boto3
from botocore.exceptions import ClientError
from common.custom_exporter import CustomConsoleSpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from common.helpers import find_span_by_type
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from monocle_apptrace.instrumentation.metamodel.botocore.handlers.botocore_span_handler import (
    BotoCoreSpanHandler,
)

logger = logging.getLogger(__name__)

AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
KNOWLEDGE_BASE_ID = os.environ.get("KNOWLEDGE_BASE_ID")
PROFILE_ARN = os.environ.get("PROFILE_ARN")

@pytest.fixture(scope="module")
def setup():
    """Setup telemetry for all tests"""
    try:
        custom_exporter = CustomConsoleSpanExporter()
        instrumentor = setup_monocle_telemetry(
            workflow_name="boto_coverage_api",
            span_processors=[SimpleSpanProcessor(custom_exporter)],
            span_handlers={"botocore_handler": BotoCoreSpanHandler()},
        )
        yield custom_exporter
    finally:
        # Clean up instrumentor to avoid global state leakage
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()

def test_retrieve_and_generate(setup):

    client = boto3.client("bedrock-agent-runtime", region_name=AWS_REGION)
    query = "What is Summary of Chapter1"

    try:
        response = client.retrieve_and_generate(
            input={
                'text': query
            },
            retrieveAndGenerateConfiguration={
                'type': 'KNOWLEDGE_BASE',
                'knowledgeBaseConfiguration': {
                    'knowledgeBaseId': KNOWLEDGE_BASE_ID,
                    'modelArn': "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
                }
            }
        )

        logger.info(f"retrieve_and_generate response: {response.get('output', {}).get('text', '')[:100]}")

        # Verify span was created
        spans = setup.get_captured_spans()
        assert len(spans) > 0, "Expected at least one span to be captured"

        # Find retrieval span
        retrieval_span = find_span_by_type(spans, "retrieval")
        assert retrieval_span is not None, "Expected to find a retrieval span"

        # Verify retrieval span attributes
        span_attrs = retrieval_span.attributes
        assert span_attrs["span.type"] == "retrieval"
        assert "entity.1.type" in span_attrs
        assert "retrieval.aws_bedrock" in span_attrs["entity.1.type"]
        assert "entity.2.name" in span_attrs
        assert "entity.2.type" in span_attrs
        assert span_attrs["entity.2.name"] == "knowledgebase"
        assert  span_attrs["entity.2.type"]=="vectorstore.knowledgebase"
        # Verify input/output events
        input_event = next((e for e in retrieval_span.events if e.name == "data.input"), None)
        output_event = next((e for e in retrieval_span.events if e.name == "data.output"), None)

        assert input_event is not None, "Expected to find data.input event"
        assert output_event is not None, "Expected to find data.output event"

        logger.info("retrieve_and_generate test passed")

    except ClientError as e:
        logger.error(f"AWS API Error: {e}")
        pytest.skip(f"Skipping due to AWS API error: {e}")


def test_invoke_model_sample(setup):
    """Test invoke_model with basic example - verifies instrumentation works"""
    setup.reset()

    bedrock_runtime = boto3.client("bedrock-runtime", region_name="eu-north-1")

    # Your question to the model
    prompt = "What is the capital city of Italy?"

    kwargs = {
        "modelId": "eu.mistral.pixtral-large-2502-v1:0",
        "contentType": "application/json",
        "accept": "application/json",
        "body": json.dumps({
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        })
    }

    # Call Bedrock
    response = bedrock_runtime.invoke_model(**kwargs)

    # Parse the response
    response_body = response['body'].read()
    output = json.loads(response_body)

    # Print the assistant's reply
    assistant_message = output['choices'][0]['message']['content']
    print("Model response:")
    print(assistant_message)
    # Verify span was created (main test goal)
    spans = setup.get_captured_spans()
    assert len(spans) > 0, "Expected at least one span to be captured"

    # Find inference span
    inference_span = find_span_by_type(spans, "inference")
    assert inference_span is not None, "Expected to find an inference span"
    span_attrs = inference_span.attributes
    assert span_attrs["span.type"] == "inference"
    assert "entity.1.type" in span_attrs
    assert span_attrs["entity.1.type"] == "inference.aws_bedrock"
    assert span_attrs["entity.1.inference_endpoint"] == "https://bedrock-runtime.eu-north-1.amazonaws.com"
    assert span_attrs["entity.1.provider_name"] == "bedrock-runtime.eu-north-1.amazonaws.com"
    assert span_attrs["entity.2.name"] == "eu.mistral.pixtral-large-2502-v1:0"
    assert span_attrs["entity.2.type"] == "model.llm.eu.mistral.pixtral-large-2502-v1:0"
    # Verify input/output events
    input_event = next((e for e in inference_span.events if e.name == "data.input"), None)
    output_event = next((e for e in inference_span.events if e.name == "data.output"), None)

    assert input_event is not None, "Expected to find data.input event"
    assert output_event is not None, "Expected to find data.output event"
    logger.info("invoke_model_sample test passed")

@pytest.mark.asyncio
async def test_invoke_data_automation_async(setup):
    setup.reset()
    runtime_client = boto3.client("bedrock-data-automation-runtime", region_name=AWS_REGION)
    response = runtime_client.invoke_data_automation_async(
        dataAutomationProfileArn=PROFILE_ARN,
        inputConfiguration={
            's3Uri': 's3://sachin-data-automation-input/chapter-1.docx'
        },
        outputConfiguration={
            's3Uri': 's3://test-sachin1/output/'
        },
        dataAutomationConfiguration={
            'dataAutomationProjectArn': 'arn:aws:bedrock:us-east-1:390041016107:data-automation-project/a224fa4c6f26',
            'stage': 'LIVE'
        }
    )

    # Log the invocation details
    invocation_arn = response.get('invocationArn')
    logger.info(f"Data automation job started: {invocation_arn}")
    logger.info(f"Initial status: {response.get('status', 'UNKNOWN')}")
    logger.info(f"  Note: This is an async job - output will be written to S3 after processing completes")
    logger.info(f" Output will be in: s3://test-sachin1/ouput/")

    # Poll job status to track progress
    try:
        max_wait_seconds = 120  # Wait up to 2 minutes
        poll_interval = 10  # Check every 10 seconds
        elapsed = 0

        while elapsed < max_wait_seconds:
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

            status_response = runtime_client.get_data_automation_status(
                invocationArn=invocation_arn
            )

            current_status = status_response.get('status', 'UNKNOWN')
            logger.info(f"  After {elapsed}s - Job status: {current_status}")

            # Log output location if available
            if 'outputConfiguration' in status_response:
                output_s3_uri = status_response['outputConfiguration'].get('s3Uri', '')
                logger.info(f" Output file location: {output_s3_uri}")

            if current_status in ['Completed', 'Success', 'Failed', 'Cancelled']:
                logger.info(f" Job finished with status: {current_status}")

                # If failed, check for error details
                if current_status == 'Failed':
                    error_msg = status_response.get('errorMessage', 'No error details available')
                    logger.error(f" Job failed: {error_msg}")

                break
        else:
            logger.warning(f"  Job still running after {max_wait_seconds}s - stopping poll")

    except Exception as e:
        logger.warning(f"Could not check job status: {e}")

    # Verify span was created
    spans = setup.get_captured_spans()
    assert len(spans) > 0, "Expected at least one span to be captured"

    inference_span = find_span_by_type(spans, "inference")
    assert inference_span is not None, "Expected to find an inference span"
    span_attrs = inference_span.attributes
    assert span_attrs["span.type"] == "inference"
    assert "entity.1.type" in span_attrs
    assert span_attrs["entity.1.type"] == "inference.aws_bedrock"
    assert span_attrs["entity.1.inference_endpoint"] == "https://bedrock-data-automation-runtime.us-east-1.amazonaws.com"
    assert span_attrs["entity.1.provider_name"] == "bedrock-data-automation-runtime.us-east-1.amazonaws.com"

    # Verify input/output events
    input_event = next((e for e in inference_span.events if e.name == "data.input"), None)
    output_event = next((e for e in inference_span.events if e.name == "data.output"), None)

    assert input_event is not None, "Expected to find data.input event"
    assert output_event is not None, "Expected to find data.output event"
    logger.info(" invoke_data_automation_async test passed (job submitted successfully)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])