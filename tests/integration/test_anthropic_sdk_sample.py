import os
import pytest
import time
import anthropic
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from common.custom_exporter import CustomConsoleSpanExporter
from monocle_apptrace.instrumentation.common.utils import logger
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
custom_exporter = CustomConsoleSpanExporter()

@pytest.fixture(autouse=True)
def clear_spans():
    """Clear spans before each test"""
    custom_exporter.reset()
    yield

@pytest.fixture(scope="module")
def setup():
    setup_monocle_telemetry(
        workflow_name="anthropic_app_1",
        span_processors=[BatchSpanProcessor(custom_exporter)],
        wrapper_methods=[
        ])

@pytest.mark.integration()
def test_anthropic_metamodel_sample(setup):
    client = anthropic.Anthropic()

    # Send a prompt to Claude
    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",  # You can use claude-3-haiku, claude-3-sonnet, etc.
        max_tokens=512,
        temperature=0.7,
        system= "You are a helpful assistant to answer questions about coffee.",
        messages=[
            {"role": "user", "content": "What is an americano?"}
        ]
    )

    # Print the response
    print("Claude's response:\n")
    print(response.content[0].text)

    time.sleep(5)
    spans = custom_exporter.get_captured_spans()

    for span in spans:
        span_attributes = span.attributes
        if span_attributes["span.type"] == "inference" or span_attributes["span.type"] == "inference.framework":
            # Assertions for all inference attributes
            assert span_attributes["entity.1.type"] == "inference.generic"
            assert span_attributes["entity.1.provider_name"] == "api.anthropic.com"
            assert span_attributes["entity.1.inference_endpoint"] == "https://api.anthropic.com"
            assert span_attributes["entity.2.name"] == "claude-3-5-sonnet-20240620"
            assert span_attributes["entity.2.type"] == "model.llm.claude-3-5-sonnet-20240620"

            # Assertions for metadata
            span_input, span_output, span_metadata = span.events
            assert "completion_tokens" in span_metadata.attributes
            assert "prompt_tokens" in span_metadata.attributes
            assert "total_tokens" in span_metadata.attributes
            
            events = span.events
            # find that there one data.input and data.output events
            assert len(events) >= 2, "Expected at least two events for input and output"
            data_input_event = [event for event in events if event.name == "data.input"][0]
            data_output_event = [event for event in events if event.name == "data.output"][0]
            assert data_input_event.name == "data.input"
            assert data_output_event.name == "data.output"
            assert "input" in data_input_event.attributes
            assert "response" in data_output_event.attributes
            assert "user" in data_input_event.attributes["input"][0]
            assert "assistant" in data_output_event.attributes["response"][0]
            # assert "system" in data_input_event.attributes["input"][0]
            # assert "You are a helpful assistant" in data_input_event.attributes["input"][0]
            assert "What is an americano?" in data_input_event.attributes["input"][0]

@pytest.mark.integration()
def test_anthropic_invalid_api_key(setup):
    try:
        client = anthropic.Anthropic(api_key="invalid_key_123")
        response = client.messages.create(
            model="claude-3-sonnet-20240620",
            max_tokens=512,
            system="You are a helpful assistant to answer questions about coffee.",
            messages=[
                {"role": "user", "content": "What is an americano?"}
            ]

        )
    except anthropic.APIError as e:
        logger.error("Authentication error: %s", str(e))

    time.sleep(5)
    spans = custom_exporter.get_captured_spans()
    for span in spans:
        if span.attributes.get("span.type") == "inference" or span.attributes.get("span.type") == "inference.framework":
            events = [e for e in span.events if e.name == "data.output"]
            assert len(events) > 0
            assert events[0].attributes["status"] == "error"
            assert "status_code" in events[0].attributes
            assert "authentication_error" in events[0].attributes.get("response", "").lower()

if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])

# {
#     "name": "anthropic.resources.messages.messages.Messages",
#     "context": {
#         "trace_id": "0x956a8a3a60ec8f6758b328cd468cbcff",
#         "span_id": "0xe9c9dd0443bce9df",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x9c16e8be5ba00c98",
#     "start_time": "2025-04-09T13:00:19.040529Z",
#     "end_time": "2025-04-09T13:00:23.930124Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.3.0",
#         "span.type": "inference",
#         "entity.1.type": "inference.generic",
#         "entity.1.provider_name": "api.anthropic.com",
#         "entity.1.inference_endpoint": "https://api.anthropic.com",
#         "entity.2.name": "claude-3-5-sonnet-20240620",
#         "entity.2.type": "model.llm.claude-3-5-sonnet-20240620",
#         "entity.count": 2
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-04-09T13:00:23.930124Z",
#             "attributes": {
#                 "input": [
#                     "{'user': 'What is an americano?'}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-04-09T13:00:23.930124Z",
#             "attributes": {
#                 "response": "{'assistant': 'An Americano is a popular coffee drink that consists of espresso diluted with hot water. Here are some key points about Americano:\n\n1. Origin: It's believed to have originated during World War II when American soldiers in Italy diluted espresso with hot water to make it similar to the coffee they were used to back home.\n\n2. Preparation: It's typically made by adding hot water to one or two shots of espresso.\n\n3. Ratio: The usual ratio is about 1:2 or 1:3 of espresso to water, but this can vary based on personal preference.\n\n4. Taste: An Americano has a similar strength to drip coffee but with a different flavor profile due to the espresso base.\n\n5. Caffeine content: It generally contains the same amount of caffeine as the espresso shots used to make it.\n\n6. Variations: It can be served hot or over ice (Iced Americano).\n\n7. Customization: Some people add milk, cream, or sweeteners to their Americano, though traditionally it's served black.\n\nAmericanos are popular among those who enjoy the flavor of espresso but prefer a larger, less intense drink compared to straight espresso shots.'}"
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-04-09T13:00:23.930124Z",
#             "attributes": {
#                 "completion_tokens": 274,
#                 "prompt_tokens": 23,
#                 "total_tokens": 297
#             }
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "anthropic_app_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "anthropic.resources.messages.messages.Messages",
#     "context": {
#         "trace_id": "0x956a8a3a60ec8f6758b328cd468cbcff",
#         "span_id": "0x9c16e8be5ba00c98",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2025-04-09T13:00:19.039530Z",
#     "end_time": "2025-04-09T13:00:23.930124Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.3.0",
#         "span.type": "workflow",
#         "entity.1.name": "anthropic_app_1",
#         "entity.1.type": "workflow.generic",
#         "entity.2.type": "app_hosting.generic",
#         "entity.2.name": "generic"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "anthropic_app_1"
#         },
#         "schema_url": ""
#     }
# }