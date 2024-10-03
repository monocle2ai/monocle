

import os

from datasets import load_dataset
from haystack import Document, Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.components.generators import OpenAIGenerator
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.retrievers.in_memory.embedding_retriever import (
    InMemoryDocumentStore,
)
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.utils import Secret
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from monocle_apptrace.instrumentor import setup_monocle_telemetry
from monocle_apptrace.wrap_common import llm_wrapper, task_wrapper
from monocle_apptrace.wrapper import WrapperMethod


def haystack_app():

    setup_monocle_telemetry(
            workflow_name="haystack_app_1",
            span_processors=[BatchSpanProcessor(ConsoleSpanExporter())],
            wrapper_methods=[
                WrapperMethod(
                    package="haystack.components.retrievers.in_memory",
                    object_name="InMemoryEmbeddingRetriever",
                    method="run",
                    span_name="haystack.retriever",
                    wrapper=llm_wrapper),

            ])

    # initialize
    api_key = os.getenv("OPENAI_API_KEY")
    generator = OpenAIGenerator(
        api_key=Secret.from_token(api_key), model="gpt-3.5-turbo"
    )

    # initialize document store, load data and store in document store
    document_store = InMemoryDocumentStore()
    dataset = load_dataset("bilgeyucel/seven-wonders", split="train")
    docs = [Document(content=doc["content"], meta=doc["meta"]) for doc in dataset]

    doc_embedder = SentenceTransformersDocumentEmbedder(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    doc_embedder.warm_up()
    docs_with_embeddings = doc_embedder.run(docs)
    document_store.write_documents(docs_with_embeddings["documents"])

    # embedder to embed user query
    text_embedder = SentenceTransformersTextEmbedder(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )

    # get relevant documents from embedded query
    retriever = InMemoryEmbeddingRetriever(document_store)

    # use documents to build the prompt
    template = """
    Given the following information, answer the question.

    Context:
    {% for document in documents %}
        {{ document.content }}
    {% endfor %}

    Question: {{question}}
    Answer:
    """

    prompt_builder = PromptBuilder(template=template)

    basic_rag_pipeline = Pipeline()
    # Add components to your pipeline
    basic_rag_pipeline.add_component("text_embedder", text_embedder)
    basic_rag_pipeline.add_component("retriever", retriever)
    basic_rag_pipeline.add_component("prompt_builder", prompt_builder)
    basic_rag_pipeline.add_component("llm", generator)

    # Now, connect the components to each other
    basic_rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    basic_rag_pipeline.connect("retriever", "prompt_builder.documents")
    basic_rag_pipeline.connect("prompt_builder", "llm")

    question = "What does Rhodes Statue look like?"

    response = basic_rag_pipeline.run(
        {"text_embedder": {"text": question}, "prompt_builder": {"question": question}}
    )

    # print(response["llm"]["replies"][0])


haystack_app()

#{
#     "name": "haystack.retriever",
#     "context": {
#         "trace_id": "0x1db120b68e3a759882ac457b07af344f",
#         "span_id": "0x88a0290f25ae392a",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x346eb05a7dd44a99",
#     "start_time": "2024-05-23T03:57:52.482232Z",
#     "end_time": "2024-05-23T03:57:52.496394Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#      "attributes": {
#         "tags": [
#             "SentenceTransformersTextEmbedder",
#             "InMemoryDocumentStore"
#         ],
#         "type": "vector_store",
#         "provider_name": "InMemoryDocumentStore",
#         "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
#     },
#     "events": [
#         {
#             "name": "context_input",
#             "timestamp": "2024-10-03T12:15:44.709938Z",
#             "attributes": {
#                 "question": "What does Rhodes Statue look like?"
#             }
#         },
#         {
#             "name": "context_output",
#             "timestamp": "2024-10-03T12:15:44.720922Z",
#             "attributes": {
#                 "response": "Within it, too, are to be seen large masses of rock, by the weight of which the artist steadied it w..."
#             }
#         }
#     ],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "haystack_app_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "haystack.openai",
#     "context": {
#         "trace_id": "0x1db120b68e3a759882ac457b07af344f",
#         "span_id": "0xc90f3343240ee785",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": "0x346eb05a7dd44a99",
#     "start_time": "2024-05-23T03:57:52.497002Z",
#     "end_time": "2024-05-23T03:57:54.793035Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "llm_input": "\n    Given the following information, answer the question.\n\n    Context:\n    \n        Within it, too, are to be seen large masses of rock, by the weight of which the artist steadied it while erecting it.[22][23]\nDestruction of the remains[edit]\nThe ultimate fate of the remains of the statue is uncertain. Rhodes has two serious earthquakes per century, owing to its location on the seismically unstable Hellenic Arc.  \n    \n\n    Question: What does Rhodes Statue look like?\n    Answer:\n    ",
#         "model_name": "gpt-3.5-turbo",
#         "completion_tokens": 90,
#         "prompt_tokens": 2464,
#         "total_tokens": 2554
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "haystack_app_1"
#         },
#         "schema_url": ""
#     }
# }
# {
#     "name": "haystack_pipeline.workflow",
#     "context": {
#         "trace_id": "0x1db120b68e3a759882ac457b07af344f",
#         "span_id": "0x346eb05a7dd44a99",
#         "trace_state": "[]"
#     },
#     "kind": "SpanKind.INTERNAL",
#     "parent_id": null,
#     "start_time": "2024-05-23T03:57:52.395585Z",
#     "end_time": "2024-05-23T03:57:54.793340Z",
#     "status": {
#         "status_code": "UNSET"
#     },
#     "attributes": {
#         "input": "What does Rhodes Statue look like?",
#         "workflow_name": "haystack_app_1",
#         "workflow_type": "workflow.haystack",
#         "output": "The Rhodes Statue, also known as the Colossus of Rhodes, depicted the Greek sun-god Helios. It was a bronze statue standing approximately 33 meters (108 feet) tall with a standard rendering of the head and face, featuring curly hair with evenly spaced spikes of bronze or silver flame radiating. The actual appearance of the rest of the statue remains unknown, but it was considered the tallest statue in the ancient world."
#     },
#     "events": [],
#     "links": [],
#     "resource": {
#         "attributes": {
#             "service.name": "haystack_app_1"
#         },
#         "schema_url": ""
#     }
# }
