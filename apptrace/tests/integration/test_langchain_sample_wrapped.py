import bs4
import pytest
from langsmith import Client
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# the method contains the untouched RAG application code
def test_langchain_sample_wrapped():

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

    result =  rag_chain.invoke("What is Task Decomposition?")


# {
#     "span_name": "langchain.task.StrOutputParser",
#     "start_time": "2024-04-16T18:44:30.033627Z",
#     "end_time": "2024-04-16T18:44:30.035181Z",
#     "duration_ms": "2",
#     "span_id": "0x098705d0420a7a40",
#     "trace_id": "0x4d297d14b25c3891eb4dd8b28453e91a",
#     "parent_id": "0x0c44185b267d8652",
#     "attributes": {},
#     "events": []
# },
# {
#     "span_name": "langchain.workflow",
#     "start_time": "2024-04-16T18:44:25.077909Z",
#     "end_time": "2024-04-16T18:44:25.442285Z",
#     "duration_ms": "364",
#     "span_id": "0x0c24a511693ca713",
#     "trace_id": "0x4d297d14b25c3891eb4dd8b28453e91a",
#     "parent_id": "0x0c44185b267d8652",
#     "attributes": {},
#     "events": []
# },
# {
#     "span_name": "langchain.workflow",
#     "start_time": "2024-04-16T18:44:24.974595Z",
#     "end_time": "2024-04-16T18:44:30.035374Z",
#     "duration_ms": "5061",
#     "span_id": "0x0c44185b267d8652",
#     "trace_id": "0x4d297d14b25c3891eb4dd8b28453e91a",
#     "parent_id": "None",
#     "attributes": {
#         "workflow_input": "What is Task Decomposition?",
#         "workflow_name": "langchain_app_1",
#         "workflow_output": "Task decomposition is a technique where complex tasks are broken down into smaller and simpler steps to enhance model performance. This process allows agents to tackle difficult tasks by transforming them into more manageable components. Task decomposition can be achieved through various methods such as using prompting techniques, task-specific instructions, or human inputs.",
#         "workflow_type": "workflow.langchain"
#     },
#     "events": []
# },
# {
#     "span_name": "langchain.task.ChatOpenAI",
#     "start_time": "2024-04-16T18:44:28.016379Z",
#     "end_time": "2024-04-16T18:44:30.033161Z",
#     "duration_ms": "2017",
#     "span_id": "0x369551685b41798f",
#     "trace_id": "0x4d297d14b25c3891eb4dd8b28453e91a",
#     "parent_id": "0x0c44185b267d8652",
#     "attributes": {
#         "model_name": "gpt-3.5-turbo-0125"
#     },
#     "events": []
# },
# {
#     "span_name": "langchain.workflow",
#     "start_time": "2024-04-16T18:44:25.080676Z",
#     "end_time": "2024-04-16T18:44:25.441839Z",
#     "duration_ms": "361",
#     "span_id": "0x7f0f48ee79169b5f",
#     "trace_id": "0x4d297d14b25c3891eb4dd8b28453e91a",
#     "parent_id": "0x0c24a511693ca713",
#     "attributes": {},
#     "events": []
# },
# {
#     "span_name": "langchain.task.ChatPromptTemplate",
#     "start_time": "2024-04-16T18:44:25.442458Z",
#     "end_time": "2024-04-16T18:44:25.443590Z",
#     "duration_ms": "1",
#     "span_id": "0xbbf8ee58d2445b42",
#     "trace_id": "0x4d297d14b25c3891eb4dd8b28453e91a",
#     "parent_id": "0x0c44185b267d8652",
#     "attributes": {},
#     "events": []
# },
# {
#     "span_name": "langchain.task.VectorStoreRetriever",
#     "start_time": "2024-04-16T18:44:25.082686Z",
#     "end_time": "2024-04-16T18:44:25.440256Z",
#     "duration_ms": "358",
#     "span_id": "0xbeb495a0888fb3f7",
#     "trace_id": "0x4d297d14b25c3891eb4dd8b28453e91a",
#     "parent_id": "0x7f0f48ee79169b5f",
#     "attributes": {},
#     "events": []
# }