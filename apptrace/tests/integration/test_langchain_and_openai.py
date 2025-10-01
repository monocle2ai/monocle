# Test multiple chains with OpenAI APIs in between
import pytest
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI

from monocle_apptrace.instrumentation.common.instrumentor import (
    setup_monocle_telemetry,
    start_trace,
    stop_trace,
)
from openai import OpenAI
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from common.custom_exporter import CustomConsoleSpanExporter
from common.chain_exec import setup_simple_chain

custom_exporter = CustomConsoleSpanExporter()


@pytest.fixture(scope="module")
def setup():
    setup_monocle_telemetry(
        workflow_name="langchain_app_1",
        span_processors=[SimpleSpanProcessor(custom_exporter)],
        wrapper_methods=[],
    )


@pytest.fixture(autouse=True)
def pre_test():
    # clear old spans
    custom_exporter.reset()


# Test multiple chains with OpenAI APIs in between. Verify each has it's workflow and inference spans
@pytest.mark.integration()
def test_langchain_with_openai(setup):
    chain1 = setup_simple_chain()
    chain2 = setup_simple_chain()
    openai = OpenAI()

    chain1.invoke("What is an americano?")
    verify_spans(
        expected_langchain_inference_spans=1,
        expected_openai_inference_spans=0,
        exptected_workflow_spans=1,
    )
    custom_exporter.reset()

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant to answer coffee related questions",
            },
            {"role": "user", "content": "What is an latte?"},
        ],
    )
    verify_spans(
        expected_langchain_inference_spans=0,
        expected_openai_inference_spans=1,
        exptected_workflow_spans=1,
    )
    custom_exporter.reset()

    chain2.invoke("What is an coffee?")
    verify_spans(
        expected_langchain_inference_spans=1,
        expected_openai_inference_spans=0,
        exptected_workflow_spans=1,
    )
    custom_exporter.reset()


# Test multiple chains with OpenAI APIs in between in a single trace Verify there only one workflow and all inference spans
@pytest.mark.integration()
def test_langchain_with_openai_single_trace(setup):
    chain1 = setup_simple_chain()
    chain2 = setup_simple_chain()
    openai = OpenAI()

    token = start_trace()
    chain1.invoke("What is an americano?")

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant to answer coffee related questions",
            },
            {"role": "user", "content": "What is an latte?"},
        ],
    )
    chain2.invoke("What is an coffee?")
    stop_trace(token)
    verify_spans(
        expected_langchain_inference_spans=2,
        expected_openai_inference_spans=1,
        exptected_workflow_spans=1,
    )


def verify_spans(
    expected_langchain_inference_spans: int,
    expected_openai_inference_spans: int,
    exptected_workflow_spans: int,
):
    spans = custom_exporter.get_captured_spans()
    workflow_spans = 0
    langchain_inference_spans = 0
    openai_inference_spans = 0
    trace_id = -1
    trace_id = spans[0].context.trace_id
    for span in spans:
        span_attributes = span.attributes
        if trace_id == -1:
            trace_id = span.context.trace_id
        else:
            assert trace_id == span.context.trace_id
        if "span.type" in span_attributes and (
            span_attributes["span.type"] == "inference"
            or span_attributes["span.type"] == "inference.framework"
        ):
            # Assertions for all inference attributes
            assert span_attributes["entity.1.type"] == "inference.openai"
            if span.name.lower().startswith("langchain"):
                langchain_inference_spans = langchain_inference_spans + 1
            elif span.name.lower().startswith("openai"):
                openai_inference_spans = openai_inference_spans + 1
        if (
            "span.type" in span_attributes
            and span_attributes["span.type"] == "workflow"
        ):
            workflow_spans = workflow_spans + 1

    assert expected_langchain_inference_spans == langchain_inference_spans
    assert expected_openai_inference_spans == openai_inference_spans
    assert exptected_workflow_spans == workflow_spans
    custom_exporter.reset()

if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])
    # pytest.main([__file__])  # Uncomment for verbose output

# {
#     "name": "langchain_core.prompts.chat.ChatPromptTemplate",
#     "context": {
#         "trace_id": "0x1089996e8645c125f945ea863ee2be8c",
#         "span_id": "0x1a886adbd4372b98",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xbfb7e47f5e0f2436",
#     "start_time": "2025-07-13T10:05:06.205736Z",
#     "end_time": "2025-07-13T10:05:06.206100Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/.venv/lib/python3.11/site-packages/langchain_core/runnables/base.py:3047",
#         "workflow.name": "langchain_app_1",
#         "span.type": "generic"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_app_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "openai.resources.chat.completions.completions.Completions",
#     "context": {
#         "trace_id": "0x1089996e8645c125f945ea863ee2be8c",
#         "span_id": "0xb8f5aa1af50b9e1c",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xe903f948c0a7bf03",
#     "start_time": "2025-07-13T10:05:06.207884Z",
#     "end_time": "2025-07-13T10:05:10.053837Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/.venv/lib/python3.11/site-packages/langchain_openai/chat_models/base.py:973",
#         "workflow.name": "langchain_app_1",
#         "span.type": "inference.modelapi"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_app_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "langchain_openai.chat_models.base.ChatOpenAI",
#     "context": {
#         "trace_id": "0x1089996e8645c125f945ea863ee2be8c",
#         "span_id": "0xe903f948c0a7bf03",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xbfb7e47f5e0f2436",
#     "start_time": "2025-07-13T10:05:06.206355Z",
#     "end_time": "2025-07-13T10:05:10.056798Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/.venv/lib/python3.11/site-packages/langchain_core/runnables/base.py:3047",
#         "workflow.name": "langchain_app_1",
#         "span.type": "inference.framework",
#         "entity.1.type": "inference.openai",
#         "entity.1.provider_name": "api.openai.com",
#         "entity.1.inference_endpoint": "https://api.openai.com/v1/",
#         "entity.2.name": "gpt-4o-mini",
#         "entity.2.type": "model.llm.gpt-4o-mini",
#         "entity.count": 2
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-07-13T10:05:10.056732Z",
#             "attributes": {
#                 "input": [
#                     "{\"system\": \"You are an assistant for question-answering tasks.If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.Respond in the same sentiment as the question\"}",
#                     "{\"human\": \"Question: What is an americano?\"}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-13T10:05:10.056774Z",
#             "attributes": {
#                 "response": "{\"ai\": \"An Americano is a type of coffee made by diluting a shot of espresso with hot water. This results in a beverage that has a similar strength to drip coffee but retains the distinct flavor of espresso. It's a popular choice for those who enjoy a milder coffee experience.\"}",
#                 "finish_reason": "stop",
#                 "finish_type": "success"
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-07-13T10:05:10.056787Z",
#             "attributes": {
#                 "temperature": 0.0,
#                 "completion_tokens": 55,
#                 "prompt_tokens": 59,
#                 "total_tokens": 114
#             }
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_app_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "langchain_core.output_parsers.string.StrOutputParser",
#     "context": {
#         "trace_id": "0x1089996e8645c125f945ea863ee2be8c",
#         "span_id": "0xb5e1624e3e51bddd",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xbfb7e47f5e0f2436",
#     "start_time": "2025-07-13T10:05:10.057434Z",
#     "end_time": "2025-07-13T10:05:10.057706Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/.venv/lib/python3.11/site-packages/langchain_core/runnables/base.py:3047",
#         "workflow.name": "langchain_app_1",
#         "span.type": "generic"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_app_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "langchain_core.runnables.base.RunnableSequence",
#     "context": {
#         "trace_id": "0x1089996e8645c125f945ea863ee2be8c",
#         "span_id": "0xbfb7e47f5e0f2436",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xad1840dbda99d56b",
#     "start_time": "2025-07-13T10:05:06.200260Z",
#     "end_time": "2025-07-13T10:05:10.057818Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/tests/integration/test_langchain_and_openai.py:43",
#         "workflow.name": "langchain_app_1",
#         "span.type": "generic"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_app_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "workflow",
#     "context": {
#         "trace_id": "0x1089996e8645c125f945ea863ee2be8c",
#         "span_id": "0xad1840dbda99d56b",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2025-07-13T10:05:06.200208Z",
#     "end_time": "2025-07-13T10:05:10.057887Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/tests/integration/test_langchain_and_openai.py:43",
#         "span.type": "workflow",
#         "entity.1.name": "langchain_app_1",
#         "entity.1.type": "workflow.langchain",
#         "entity.2.type": "app_hosting.generic",
#         "entity.2.name": "generic"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_app_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "openai.resources.chat.completions.completions.Completions",
#     "context": {
#         "trace_id": "0x4f0b8f8fe0a493cf74d75443d9bf0820",
#         "span_id": "0x51dcb1b567ac0835",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x9b63405676687341",
#     "start_time": "2025-07-13T10:05:10.058217Z",
#     "end_time": "2025-07-13T10:05:12.202089Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/tests/integration/test_langchain_and_openai.py:51",
#         "workflow.name": "langchain_app_1",
#         "span.type": "inference",
#         "entity.1.type": "inference.openai",
#         "entity.1.provider_name": "api.openai.com",
#         "entity.1.inference_endpoint": "https://api.openai.com/v1/",
#         "entity.2.name": "gpt-4o-mini",
#         "entity.2.type": "model.llm.gpt-4o-mini",
#         "entity.count": 2
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-07-13T10:05:12.202043Z",
#             "attributes": {
#                 "input": [
#                     "{\"system\": \"You are a helpful assistant to answer coffee related questions\"}",
#                     "{\"user\": \"What is an latte?\"}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-13T10:05:12.202071Z",
#             "attributes": {
#                 "response": "{\"assistant\": \"A latte, short for \\\"caff\\u00e8 latte,\\\" is an Italian coffee beverage made with espresso and steamed milk. It typically consists of one or two shots of espresso, topped with a significant amount of steamed milk, and finished with a small layer of milk foam. The ratio of milk to espresso in a latte is generally about 3:1, which gives it a creamy texture and a mild coffee flavor. Lattes can be flavored with syrups, such as vanilla or caramel, and are often enjoyed hot but can also be served iced.\"}",
#                 "finish_reason": "stop",
#                 "finish_type": "success"
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-07-13T10:05:12.202080Z",
#             "attributes": {
#                 "completion_tokens": 109,
#                 "prompt_tokens": 26,
#                 "total_tokens": 135
#             }
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_app_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "workflow",
#     "context": {
#         "trace_id": "0x4f0b8f8fe0a493cf74d75443d9bf0820",
#         "span_id": "0x9b63405676687341",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2025-07-13T10:05:10.058155Z",
#     "end_time": "2025-07-13T10:05:12.202414Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/tests/integration/test_langchain_and_openai.py:51",
#         "span.type": "workflow",
#         "entity.1.name": "langchain_app_1",
#         "entity.1.type": "workflow.generic",
#         "entity.2.type": "app_hosting.generic",
#         "entity.2.name": "generic"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_app_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "langchain_core.prompts.chat.ChatPromptTemplate",
#     "context": {
#         "trace_id": "0xc003aad6180428487171a9543b14ca5f",
#         "span_id": "0xc3e44c9be714b9e2",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xea8d0ebae16b48c4",
#     "start_time": "2025-07-13T10:05:12.203380Z",
#     "end_time": "2025-07-13T10:05:12.203717Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/.venv/lib/python3.11/site-packages/langchain_core/runnables/base.py:3047",
#         "workflow.name": "langchain_app_1",
#         "span.type": "generic"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_app_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "openai.resources.chat.completions.completions.Completions",
#     "context": {
#         "trace_id": "0xc003aad6180428487171a9543b14ca5f",
#         "span_id": "0xcbf7f5102912d756",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xa37dfb808527faf5",
#     "start_time": "2025-07-13T10:05:12.204345Z",
#     "end_time": "2025-07-13T10:05:13.635185Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/.venv/lib/python3.11/site-packages/langchain_openai/chat_models/base.py:973",
#         "workflow.name": "langchain_app_1",
#         "span.type": "inference.modelapi"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_app_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "langchain_openai.chat_models.base.ChatOpenAI",
#     "context": {
#         "trace_id": "0xc003aad6180428487171a9543b14ca5f",
#         "span_id": "0xa37dfb808527faf5",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xea8d0ebae16b48c4",
#     "start_time": "2025-07-13T10:05:12.203952Z",
#     "end_time": "2025-07-13T10:05:13.635760Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/.venv/lib/python3.11/site-packages/langchain_core/runnables/base.py:3047",
#         "workflow.name": "langchain_app_1",
#         "span.type": "inference.framework",
#         "entity.1.type": "inference.openai",
#         "entity.1.provider_name": "api.openai.com",
#         "entity.1.inference_endpoint": "https://api.openai.com/v1/",
#         "entity.2.name": "gpt-4o-mini",
#         "entity.2.type": "model.llm.gpt-4o-mini",
#         "entity.count": 2
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-07-13T10:05:13.635676Z",
#             "attributes": {
#                 "input": [
#                     "{\"system\": \"You are an assistant for question-answering tasks.If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.Respond in the same sentiment as the question\"}",
#                     "{\"human\": \"Question: What is an coffee?\"}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-13T10:05:13.635728Z",
#             "attributes": {
#                 "response": "{\"ai\": \"Coffee is a popular beverage made from roasted coffee beans, which are the seeds of the Coffea plant. It is typically brewed with hot water and can be enjoyed in various forms, such as espresso, cappuccino, or iced coffee. Many people drink coffee for its rich flavor and stimulating effects due to caffeine.\"}",
#                 "finish_reason": "stop",
#                 "finish_type": "success"
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-07-13T10:05:13.635750Z",
#             "attributes": {
#                 "temperature": 0.0,
#                 "completion_tokens": 62,
#                 "prompt_tokens": 59,
#                 "total_tokens": 121
#             }
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_app_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "langchain_core.output_parsers.string.StrOutputParser",
#     "context": {
#         "trace_id": "0xc003aad6180428487171a9543b14ca5f",
#         "span_id": "0x2666a4267469fe19",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xea8d0ebae16b48c4",
#     "start_time": "2025-07-13T10:05:13.636393Z",
#     "end_time": "2025-07-13T10:05:13.636628Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/.venv/lib/python3.11/site-packages/langchain_core/runnables/base.py:3047",
#         "workflow.name": "langchain_app_1",
#         "span.type": "generic"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_app_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "langchain_core.runnables.base.RunnableSequence",
#     "context": {
#         "trace_id": "0xc003aad6180428487171a9543b14ca5f",
#         "span_id": "0xea8d0ebae16b48c4",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xc9dba016ca69b33d",
#     "start_time": "2025-07-13T10:05:12.202767Z",
#     "end_time": "2025-07-13T10:05:13.636755Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/tests/integration/test_langchain_and_openai.py:68",
#         "workflow.name": "langchain_app_1",
#         "span.type": "generic"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_app_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "workflow",
#     "context": {
#         "trace_id": "0xc003aad6180428487171a9543b14ca5f",
#         "span_id": "0xc9dba016ca69b33d",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2025-07-13T10:05:12.202731Z",
#     "end_time": "2025-07-13T10:05:13.636841Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/tests/integration/test_langchain_and_openai.py:68",
#         "span.type": "workflow",
#         "entity.1.name": "langchain_app_1",
#         "entity.1.type": "workflow.langchain",
#         "entity.2.type": "app_hosting.generic",
#         "entity.2.name": "generic"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_app_1"
#         },
#         "schema_url": ""
#     }
# }
# .{
#     "name": "langchain_core.prompts.chat.ChatPromptTemplate",
#     "context": {
#         "trace_id": "0x89d6b54ad6ddb2da55f1ac7c298d0ab4",
#         "span_id": "0x4a3592e193c5789f",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x7215991e17c24c4d",
#     "start_time": "2025-07-13T10:05:13.674047Z",
#     "end_time": "2025-07-13T10:05:13.674343Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/.venv/lib/python3.11/site-packages/langchain_core/runnables/base.py:3047",
#         "workflow.name": "langchain_app_1",
#         "span.type": "generic"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_app_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "openai.resources.chat.completions.completions.Completions",
#     "context": {
#         "trace_id": "0x89d6b54ad6ddb2da55f1ac7c298d0ab4",
#         "span_id": "0xe10b8b09cffa4973",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0xa85c3fac52aaf19a",
#     "start_time": "2025-07-13T10:05:13.674976Z",
#     "end_time": "2025-07-13T10:05:14.912076Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/.venv/lib/python3.11/site-packages/langchain_openai/chat_models/base.py:973",
#         "workflow.name": "langchain_app_1",
#         "span.type": "inference.modelapi"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_app_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "langchain_openai.chat_models.base.ChatOpenAI",
#     "context": {
#         "trace_id": "0x89d6b54ad6ddb2da55f1ac7c298d0ab4",
#         "span_id": "0xa85c3fac52aaf19a",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x7215991e17c24c4d",
#     "start_time": "2025-07-13T10:05:13.674589Z",
#     "end_time": "2025-07-13T10:05:14.912711Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/.venv/lib/python3.11/site-packages/langchain_core/runnables/base.py:3047",
#         "workflow.name": "langchain_app_1",
#         "span.type": "inference.framework",
#         "entity.1.type": "inference.openai",
#         "entity.1.provider_name": "api.openai.com",
#         "entity.1.inference_endpoint": "https://api.openai.com/v1/",
#         "entity.2.name": "gpt-4o-mini",
#         "entity.2.type": "model.llm.gpt-4o-mini",
#         "entity.count": 2
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-07-13T10:05:14.912627Z",
#             "attributes": {
#                 "input": [
#                     "{\"system\": \"You are an assistant for question-answering tasks.If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.Respond in the same sentiment as the question\"}",
#                     "{\"human\": \"Question: What is an americano?\"}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-13T10:05:14.912680Z",
#             "attributes": {
#                 "response": "{\"ai\": \"An Americano is a type of coffee made by diluting espresso with hot water. This results in a drink that has a similar strength to brewed coffee but retains the distinct flavor of espresso. It's a popular choice for those who enjoy a milder coffee experience.\"}",
#                 "finish_reason": "stop",
#                 "finish_type": "success"
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-07-13T10:05:14.912700Z",
#             "attributes": {
#                 "temperature": 0.0,
#                 "completion_tokens": 52,
#                 "prompt_tokens": 59,
#                 "total_tokens": 111
#             }
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_app_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "langchain_core.output_parsers.string.StrOutputParser",
#     "context": {
#         "trace_id": "0x89d6b54ad6ddb2da55f1ac7c298d0ab4",
#         "span_id": "0xb1f433655fc794a8",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x7215991e17c24c4d",
#     "start_time": "2025-07-13T10:05:14.913535Z",
#     "end_time": "2025-07-13T10:05:14.913851Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/.venv/lib/python3.11/site-packages/langchain_core/runnables/base.py:3047",
#         "workflow.name": "langchain_app_1",
#         "span.type": "generic"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_app_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "langchain_core.runnables.base.RunnableSequence",
#     "context": {
#         "trace_id": "0x89d6b54ad6ddb2da55f1ac7c298d0ab4",
#         "span_id": "0x7215991e17c24c4d",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x2af9a92c5b0944cb",
#     "start_time": "2025-07-13T10:05:13.673387Z",
#     "end_time": "2025-07-13T10:05:14.914022Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/tests/integration/test_langchain_and_openai.py:85",
#         "workflow.name": "langchain_app_1",
#         "span.type": "generic"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_app_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "openai.resources.chat.completions.completions.Completions",
#     "context": {
#         "trace_id": "0x89d6b54ad6ddb2da55f1ac7c298d0ab4",
#         "span_id": "0xd6fb71b7219bf393",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x2af9a92c5b0944cb",
#     "start_time": "2025-07-13T10:05:14.914447Z",
#     "end_time": "2025-07-13T10:05:17.589018Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/tests/integration/test_langchain_and_openai.py:87",
#         "workflow.name": "langchain_app_1",
#         "span.type": "inference",
#         "entity.1.type": "inference.openai",
#         "entity.1.provider_name": "api.openai.com",
#         "entity.1.inference_endpoint": "https://api.openai.com/v1/",
#         "entity.2.name": "gpt-4o-mini",
#         "entity.2.type": "model.llm.gpt-4o-mini",
#         "entity.count": 2
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-07-13T10:05:17.588977Z",
#             "attributes": {
#                 "input": [
#                     "{\"system\": \"You are a helpful assistant to answer coffee related questions\"}",
#                     "{\"user\": \"What is an latte?\"}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-13T10:05:17.589002Z",
#             "attributes": {
#                 "response": "{\"assistant\": \"A latte, short for \\\"caff\\u00e8 latte,\\\" is a popular coffee beverage made with espresso and steamed milk. Typically, a latte consists of one or two shots of espresso, which is then combined with a larger amount of steamed milk, and usually topped with a small amount of milk foam. The ratio often used is about one part espresso to three parts steamed milk, though this can vary according to personal preference. A latte can be served hot or iced and is sometimes flavored with syrups such as vanilla or caramel. It's known for its creamy texture and mild coffee flavor, making it a favorite for many coffee drinkers.\"}",
#                 "finish_reason": "stop",
#                 "finish_type": "success"
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-07-13T10:05:17.589010Z",
#             "attributes": {
#                 "completion_tokens": 125,
#                 "prompt_tokens": 26,
#                 "total_tokens": 151
#             }
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_app_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "langchain_core.prompts.chat.ChatPromptTemplate",
#     "context": {
#         "trace_id": "0x89d6b54ad6ddb2da55f1ac7c298d0ab4",
#         "span_id": "0xb5f11d5fd95608ad",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x073f38c186020c88",
#     "start_time": "2025-07-13T10:05:17.590098Z",
#     "end_time": "2025-07-13T10:05:17.590383Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/.venv/lib/python3.11/site-packages/langchain_core/runnables/base.py:3047",
#         "workflow.name": "langchain_app_1",
#         "span.type": "generic"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_app_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "openai.resources.chat.completions.completions.Completions",
#     "context": {
#         "trace_id": "0x89d6b54ad6ddb2da55f1ac7c298d0ab4",
#         "span_id": "0x8a213cb34aa00685",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x83c68548a67ba85f",
#     "start_time": "2025-07-13T10:05:17.590937Z",
#     "end_time": "2025-07-13T10:05:19.702694Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/.venv/lib/python3.11/site-packages/langchain_openai/chat_models/base.py:973",
#         "workflow.name": "langchain_app_1",
#         "span.type": "inference.modelapi"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_app_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "langchain_openai.chat_models.base.ChatOpenAI",
#     "context": {
#         "trace_id": "0x89d6b54ad6ddb2da55f1ac7c298d0ab4",
#         "span_id": "0x83c68548a67ba85f",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x073f38c186020c88",
#     "start_time": "2025-07-13T10:05:17.590587Z",
#     "end_time": "2025-07-13T10:05:19.702987Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/.venv/lib/python3.11/site-packages/langchain_core/runnables/base.py:3047",
#         "workflow.name": "langchain_app_1",
#         "span.type": "inference.framework",
#         "entity.1.type": "inference.openai",
#         "entity.1.provider_name": "api.openai.com",
#         "entity.1.inference_endpoint": "https://api.openai.com/v1/",
#         "entity.2.name": "gpt-4o-mini",
#         "entity.2.type": "model.llm.gpt-4o-mini",
#         "entity.count": 2
#     },
#     "events": [
#         {
#             "name": "data.input",
#             "timestamp": "2025-07-13T10:05:19.702951Z",
#             "attributes": {
#                 "input": [
#                     "{\"system\": \"You are an assistant for question-answering tasks.If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.Respond in the same sentiment as the question\"}",
#                     "{\"human\": \"Question: What is an coffee?\"}"
#                 ]
#             }
#         },
#         {
#             "name": "data.output",
#             "timestamp": "2025-07-13T10:05:19.702974Z",
#             "attributes": {
#                 "response": "{\"ai\": \"Coffee is a popular beverage made from roasted coffee beans, which are the seeds of the Coffea plant. It is typically brewed with hot water and can be enjoyed in various forms, such as espresso, drip coffee, or cold brew. Many people drink coffee for its rich flavor and stimulating effects due to caffeine.\"}",
#                 "finish_reason": "stop",
#                 "finish_type": "success"
#             }
#         },
#         {
#             "name": "metadata",
#             "timestamp": "2025-07-13T10:05:19.702983Z",
#             "attributes": {
#                 "temperature": 0.0,
#                 "completion_tokens": 62,
#                 "prompt_tokens": 59,
#                 "total_tokens": 121
#             }
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_app_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "langchain_core.output_parsers.string.StrOutputParser",
#     "context": {
#         "trace_id": "0x89d6b54ad6ddb2da55f1ac7c298d0ab4",
#         "span_id": "0x937b3376fd7cba1d",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x073f38c186020c88",
#     "start_time": "2025-07-13T10:05:19.703355Z",
#     "end_time": "2025-07-13T10:05:19.703498Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/.venv/lib/python3.11/site-packages/langchain_core/runnables/base.py:3047",
#         "workflow.name": "langchain_app_1",
#         "span.type": "generic"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_app_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "langchain_core.runnables.base.RunnableSequence",
#     "context": {
#         "trace_id": "0x89d6b54ad6ddb2da55f1ac7c298d0ab4",
#         "span_id": "0x073f38c186020c88",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x2af9a92c5b0944cb",
#     "start_time": "2025-07-13T10:05:17.589543Z",
#     "end_time": "2025-07-13T10:05:19.703587Z",
#     "status": {
#         "status_code": "OK"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "/Users/kshitizvijayvargiya/monocle-ksh/tests/integration/test_langchain_and_openai.py:97",
#         "workflow.name": "langchain_app_1",
#         "span.type": "generic"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_app_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "workflow",
#     "context": {
#         "trace_id": "0x89d6b54ad6ddb2da55f1ac7c298d0ab4",
#         "span_id": "0x2af9a92c5b0944cb",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2025-07-13T10:05:13.673187Z",
#     "end_time": "2025-07-13T10:05:19.703641Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "monocle_apptrace.version": "0.4.0",
#         "monocle_apptrace.language": "python",
#         "span_source": "",
#         "span.type": "workflow",
#         "entity.1.name": "langchain_app_1",
#         "entity.1.type": "workflow.generic",
#         "entity.2.type": "app_hosting.generic",
#         "entity.2.name": "generic"
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "langchain_app_1"
#         },
#         "schema_url": ""
#     }
# }
