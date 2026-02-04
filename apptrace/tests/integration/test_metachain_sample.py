import logging

import bs4
import pytest
from common.langhchain_patch import create_history_aware_retriever
from dotenv import load_dotenv
from langsmith import Client
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from metamodel.metachain.methods import METACHAIN_METHODS
from monocle_apptrace.instrumentation.common.instrumentor import (
    set_context_properties,
    setup_monocle_telemetry,
)
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

logger = logging.getLogger(__name__)

@pytest.fixture(scope="module")
def setup():
    try:
        logging.basicConfig(level=logging.INFO)
        load_dotenv()
        instrumentor = setup_monocle_telemetry(
                    workflow_name="langchain_app_1",
                    span_processors=[BatchSpanProcessor(ConsoleSpanExporter())],
                    wrapper_methods=METACHAIN_METHODS, union_with_default_methods=True)
        yield
    finally:
        # Clean up instrumentor to avoid global state leakage
        if instrumentor and instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.uninstrument()

def test_metachain_sample(setup):
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

    # Load, chunk and index the contents of the blog.
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

    # Retrieve and generate using the relevant snippets of the blog.
    retriever = vectorstore.as_retriever()
    client = Client()
    prompt = client.pull_prompt("rlm/rag-prompt")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Use three sentences maximum and keep the answer concise.\

    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )


    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    chat_history = []

    set_context_properties({"session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16"})

    question = "What is Task Decomposition?"
    ai_msg_1 = rag_chain.invoke({"input": question, "chat_history": chat_history})
    logger.info(ai_msg_1["answer"])
    chat_history.extend([HumanMessage(content=question), ai_msg_1["answer"]])

    second_question = "What are common ways of doing it?"
    ai_msg_2 = rag_chain.invoke({"input": second_question, "chat_history": chat_history})

    logger.info(ai_msg_2["answer"])


#ndjson format stored in s3_bucket

# {"name": "langchain.task.ChatOpenAI", "context": {"trace_id": "0x5b964bc8323611c33bedfb2ba1c02297", "span_id": "0xef1a5270c100927d", "trace_state": "[]"}, "kind": "SpanKind.INTERNAL", "parent_id": "0x0ea09995ad209078", "start_time": "2024-10-22T06:29:20.705616Z", "end_time": "2024-10-22T06:29:22.488604Z", "status": {"status_code": "UNSET"}, "attributes": {"session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16", "span.type": "inference", "entity.count": 2, "entity.1.type": "inference.azure_openai", "entity.1.provider_name": "api.openai.com", "entity.2.name": "gpt-3.5-turbo-0125", "entity.2.type": "model.llm.gpt-3.5-turbo-0125"}, "events": [{"name": "metadata", "timestamp": "2024-10-22T06:29:22.488587Z", "attributes": {"temperature": 0.7, "completion_tokens": 82, "prompt_tokens": 580, "total_tokens": 662}}], "links": [], "resource": {"attributes": {"service.name": "langchain_app_1"}, "schema_url": ""}}
# {"name": "langchain.task.StrOutputParser", "context": {"trace_id": "0x5b964bc8323611c33bedfb2ba1c02297", "span_id": "0x4ffc0cd2351f4560", "trace_state": "[]"}, "kind": "SpanKind.INTERNAL", "parent_id": "0x0ea09995ad209078", "start_time": "2024-10-22T06:29:22.488731Z", "end_time": "2024-10-22T06:29:22.488930Z", "status": {"status_code": "UNSET"}, "attributes": {"session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16"}, "events": [], "links": [], "resource": {"attributes": {"service.name": "langchain_app_1"}, "schema_url": ""}}
#