

import bs4
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from monocle_apptrace.instrumentor import set_context_properties, setup_monocle_telemetry
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from langhchain_patch import create_history_aware_retriever
from monocle_apptrace.exporters.aws.s3_exporter import S3SpanExporter
import logging
logging.basicConfig(level=logging.INFO)
import os
from dotenv import load_dotenv, dotenv_values
load_dotenv()
os.environ["OPENAI_API_KEY"] = ""
os.environ['AWS_ACCESS_KEY_ID'] = ''
os.environ['AWS_SECRET_ACCESS_KEY'] = ''
exporter = S3SpanExporter(
    region_name='us-east-1',
    bucket_name='sachin-dev'
)
setup_monocle_telemetry(
            workflow_name="langchain_app_1",
            span_processors=[BatchSpanProcessor(exporter)],
            wrapper_methods=[])

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
prompt = hub.pull("rlm/rag-prompt")

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
print(ai_msg_1["answer"])
chat_history.extend([HumanMessage(content=question), ai_msg_1["answer"]])

second_question = "What are common ways of doing it?"
ai_msg_2 = rag_chain.invoke({"input": second_question, "chat_history": chat_history})

print(ai_msg_2["answer"])


#ndjson format stored in s3_bucket

# {"name": "langchain.task.ChatOpenAI", "context": {"trace_id": "0x5b964bc8323611c33bedfb2ba1c02297", "span_id": "0xef1a5270c100927d", "trace_state": "[]"}, "kind": "SpanKind.INTERNAL", "parent_id": "0x0ea09995ad209078", "start_time": "2024-10-22T06:29:20.705616Z", "end_time": "2024-10-22T06:29:22.488604Z", "status": {"status_code": "UNSET"}, "attributes": {"session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16", "span.type": "inference", "entity.count": 2, "entity.1.type": "inference.azure_oai", "entity.1.provider_name": "api.openai.com", "entity.2.name": "gpt-3.5-turbo-0125", "entity.2.type": "model.llm.gpt-3.5-turbo-0125"}, "events": [{"name": "metadata", "timestamp": "2024-10-22T06:29:22.488587Z", "attributes": {"temperature": 0.7, "completion_tokens": 82, "prompt_tokens": 580, "total_tokens": 662}}], "links": [], "resource": {"attributes": {"service.name": "langchain_app_1"}, "schema_url": ""}}
# {"name": "langchain.task.StrOutputParser", "context": {"trace_id": "0x5b964bc8323611c33bedfb2ba1c02297", "span_id": "0x4ffc0cd2351f4560", "trace_state": "[]"}, "kind": "SpanKind.INTERNAL", "parent_id": "0x0ea09995ad209078", "start_time": "2024-10-22T06:29:22.488731Z", "end_time": "2024-10-22T06:29:22.488930Z", "status": {"status_code": "UNSET"}, "attributes": {"session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16"}, "events": [], "links": [], "resource": {"attributes": {"service.name": "langchain_app_1"}, "schema_url": ""}}
# {"name": "langchain.workflow", "context": {"trace_id": "0x5b964bc8323611c33bedfb2ba1c02297", "span_id": "0x0ea09995ad209078", "trace_state": "[]"}, "kind": "SpanKind.INTERNAL", "parent_id": "0xd68372685da81f95", "start_time": "2024-10-22T06:29:20.699815Z", "end_time": "2024-10-22T06:29:22.488947Z", "status": {"status_code": "UNSET"}, "attributes": {"session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16"}, "events": [], "links": [], "resource": {"attributes": {"service.name": "langchain_app_1"}, "schema_url": ""}}
# {"name": "langchain.workflow", "context": {"trace_id": "0x5b964bc8323611c33bedfb2ba1c02297", "span_id": "0xd68372685da81f95", "trace_state": "[]"}, "kind": "SpanKind.INTERNAL", "parent_id": "0xf91b4645d4d87e4f", "start_time": "2024-10-22T06:29:20.698849Z", "end_time": "2024-10-22T06:29:22.489223Z", "status": {"status_code": "UNSET"}, "attributes": {"session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16"}, "events": [], "links": [], "resource": {"attributes": {"service.name": "langchain_app_1"}, "schema_url": ""}}
# {"name": "langchain.workflow", "context": {"trace_id": "0x5b964bc8323611c33bedfb2ba1c02297", "span_id": "0xf91b4645d4d87e4f", "trace_state": "[]"}, "kind": "SpanKind.INTERNAL", "parent_id": null, "start_time": "2024-10-22T06:29:20.216357Z", "end_time": "2024-10-22T06:29:22.489346Z", "status": {"status_code": "UNSET"}, "attributes": {"session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16", "workflow_name": "langchain_app_1", "workflow_type": "workflow.langchain"}, "events": [{"name": "data.input", "timestamp": "2024-10-22T06:29:20.216407Z", "attributes": {"input": "What is Task Decomposition?", "chat_history": []}}, {"name": "data.output", "timestamp": "2024-10-22T06:29:22.489338Z", "attributes": {"input": "What is Task Decomposition?", "chat_history": [], "answer": "Task decomposition is a technique used to break down complex tasks into smaller and simpler steps, making them more manageable for agents or models to handle. This process involves transforming big tasks into multiple subtasks, allowing for a more detailed and structured approach to problem-solving. Task decomposition can be implemented through various methods, such as using prompting techniques like Chain of Thought or Tree of Thoughts, task-specific instructions, or human inputs."}}], "links": [], "resource": {"attributes": {"service.name": "langchain_app_1"}, "schema_url": ""}}
# {"name": "langchain.task.ChatPromptTemplate", "context": {"trace_id": "0xcb7359e268252dc33578c7cea5849bb9", "span_id": "0x05eaed3c7e842195", "trace_state": "[]"}, "kind": "SpanKind.INTERNAL", "parent_id": "0x2e24a535f93c42a9", "start_time": "2024-10-22T06:29:22.491381Z", "end_time": "2024-10-22T06:29:22.492085Z", "status": {"status_code": "UNSET"}, "attributes": {"session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16"}, "events": [], "links": [], "resource": {"attributes": {"service.name": "langchain_app_1"}, "schema_url": ""}}
# {"name": "langchain.task.ChatOpenAI", "context": {"trace_id": "0xcb7359e268252dc33578c7cea5849bb9", "span_id": "0x25c2952a32cd38f4", "trace_state": "[]"}, "kind": "SpanKind.INTERNAL", "parent_id": "0x2e24a535f93c42a9", "start_time": "2024-10-22T06:29:22.492179Z", "end_time": "2024-10-22T06:29:23.279336Z", "status": {"status_code": "UNSET"}, "attributes": {"session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16", "span.type": "inference", "entity.count": 2, "entity.1.type": "inference.azure_oai", "entity.1.provider_name": "api.openai.com", "entity.2.name": "gpt-3.5-turbo-0125", "entity.2.type": "model.llm.gpt-3.5-turbo-0125"}, "events": [{"name": "metadata", "timestamp": "2024-10-22T06:29:23.279302Z", "attributes": {"temperature": 0.7, "completion_tokens": 9, "prompt_tokens": 165, "total_tokens": 174}}], "links": [], "resource": {"attributes": {"service.name": "langchain_app_1"}, "schema_url": ""}}
# {"name": "langchain.task.StrOutputParser", "context": {"trace_id": "0xcb7359e268252dc33578c7cea5849bb9", "span_id": "0x225df564d78acc8e", "trace_state": "[]"}, "kind": "SpanKind.INTERNAL", "parent_id": "0x2e24a535f93c42a9", "start_time": "2024-10-22T06:29:23.279658Z", "end_time": "2024-10-22T06:29:23.280195Z", "status": {"status_code": "UNSET"}, "attributes": {"session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16"}, "events": [], "links": [], "resource": {"attributes": {"service.name": "langchain_app_1"}, "schema_url": ""}}
# {"name": "langchain.task.VectorStoreRetriever", "context": {"trace_id": "0xcb7359e268252dc33578c7cea5849bb9", "span_id": "0x5cecb3567255cace", "trace_state": "[]"}, "kind": "SpanKind.INTERNAL", "parent_id": "0x2e24a535f93c42a9", "start_time": "2024-10-22T06:29:23.280410Z", "end_time": "2024-10-22T06:29:23.761006Z", "status": {"status_code": "UNSET"}, "attributes": {"session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16", "span.type": "retrieval", "entity.count": 2, "entity.1.name": "Chroma", "entity.1.type": "vectorstore.Chroma", "entity.2.name": "text-embedding-ada-002", "entity.2.type": "model.embedding.text-embedding-ada-002"}, "events": [{"name": "data.input", "timestamp": "2024-10-22T06:29:23.280792Z", "attributes": {"question": "What are some typical methods for task decomposition?"}}, {"name": "data.output", "timestamp": "2024-10-22T06:29:23.760974Z", "attributes": {"response": "Tree of Thoughts (Yao et al. 2023) extends CoT by exploring multiple reasoning possibilities at each..."}}], "links": [], "resource": {"attributes": {"service.name": "langchain_app_1"}, "schema_url": ""}}
# {"name": "langchain.workflow", "context": {"trace_id": "0xcb7359e268252dc33578c7cea5849bb9", "span_id": "0x2e24a535f93c42a9", "trace_state": "[]"}, "kind": "SpanKind.INTERNAL", "parent_id": "0x548cb0b11c2cea74", "start_time": "2024-10-22T06:29:22.491196Z", "end_time": "2024-10-22T06:29:23.761073Z", "status": {"status_code": "UNSET"}, "attributes": {"session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16"}, "events": [], "links": [], "resource": {"attributes": {"service.name": "langchain_app_1"}, "schema_url": ""}}
# {"name": "langchain.workflow", "context": {"trace_id": "0xcb7359e268252dc33578c7cea5849bb9", "span_id": "0x548cb0b11c2cea74", "trace_state": "[]"}, "kind": "SpanKind.INTERNAL", "parent_id": "0x7876cedb29936d57", "start_time": "2024-10-22T06:29:22.490453Z", "end_time": "2024-10-22T06:29:23.761486Z", "status": {"status_code": "UNSET"}, "attributes": {"session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16"}, "events": [], "links": [], "resource": {"attributes": {"service.name": "langchain_app_1"}, "schema_url": ""}}
# {"name": "langchain.workflow", "context": {"trace_id": "0xcb7359e268252dc33578c7cea5849bb9", "span_id": "0xda5ae5009d296c6a", "trace_state": "[]"}, "kind": "SpanKind.INTERNAL", "parent_id": "0x51082b49084e96f7", "start_time": "2024-10-22T06:29:23.765467Z", "end_time": "2024-10-22T06:29:23.767042Z", "status": {"status_code": "UNSET"}, "attributes": {"session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16"}, "events": [], "links": [], "resource": {"attributes": {"service.name": "langchain_app_1"}, "schema_url": ""}}
# {"name": "langchain.task.ChatPromptTemplate", "context": {"trace_id": "0xcb7359e268252dc33578c7cea5849bb9", "span_id": "0x3ba5c057d613a76e", "trace_state": "[]"}, "kind": "SpanKind.INTERNAL", "parent_id": "0x51082b49084e96f7", "start_time": "2024-10-22T06:29:23.767278Z", "end_time": "2024-10-22T06:29:23.768807Z", "status": {"status_code": "UNSET"}, "attributes": {"session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16"}, "events": [], "links": [], "resource": {"attributes": {"service.name": "langchain_app_1"}, "schema_url": ""}}
# {"name": "langchain.task.ChatOpenAI", "context": {"trace_id": "0xcb7359e268252dc33578c7cea5849bb9", "span_id": "0xecb8328d1f850357", "trace_state": "[]"}, "kind": "SpanKind.INTERNAL", "parent_id": "0x51082b49084e96f7", "start_time": "2024-10-22T06:29:23.769035Z", "end_time": "2024-10-22T06:29:25.227443Z", "status": {"status_code": "UNSET"}, "attributes": {"session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16", "span.type": "inference", "entity.count": 2, "entity.1.type": "inference.azure_oai", "entity.1.provider_name": "api.openai.com", "entity.2.name": "gpt-3.5-turbo-0125", "entity.2.type": "model.llm.gpt-3.5-turbo-0125"}, "events": [{"name": "metadata", "timestamp": "2024-10-22T06:29:25.227427Z", "attributes": {"temperature": 0.7, "completion_tokens": 82, "prompt_tokens": 678, "total_tokens": 760}}], "links": [], "resource": {"attributes": {"service.name": "langchain_app_1"}, "schema_url": ""}}
# {"name": "langchain.task.StrOutputParser", "context": {"trace_id": "0xcb7359e268252dc33578c7cea5849bb9", "span_id": "0x64820fe1cc9ff4ac", "trace_state": "[]"}, "kind": "SpanKind.INTERNAL", "parent_id": "0x51082b49084e96f7", "start_time": "2024-10-22T06:29:25.227565Z", "end_time": "2024-10-22T06:29:25.227777Z", "status": {"status_code": "UNSET"}, "attributes": {"session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16"}, "events": [], "links": [], "resource": {"attributes": {"service.name": "langchain_app_1"}, "schema_url": ""}}
# {"name": "langchain.workflow", "context": {"trace_id": "0xcb7359e268252dc33578c7cea5849bb9", "span_id": "0x51082b49084e96f7", "trace_state": "[]"}, "kind": "SpanKind.INTERNAL", "parent_id": "0xe2ad7a2d01df54e9", "start_time": "2024-10-22T06:29:23.763336Z", "end_time": "2024-10-22T06:29:25.227795Z", "status": {"status_code": "UNSET"}, "attributes": {"session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16"}, "events": [], "links": [], "resource": {"attributes": {"service.name": "langchain_app_1"}, "schema_url": ""}}
# {"name": "langchain.workflow", "context": {"trace_id": "0xcb7359e268252dc33578c7cea5849bb9", "span_id": "0xe2ad7a2d01df54e9", "trace_state": "[]"}, "kind": "SpanKind.INTERNAL", "parent_id": "0x7876cedb29936d57", "start_time": "2024-10-22T06:29:23.762321Z", "end_time": "2024-10-22T06:29:25.228051Z", "status": {"status_code": "UNSET"}, "attributes": {"session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16"}, "events": [], "links": [], "resource": {"attributes": {"service.name": "langchain_app_1"}, "schema_url": ""}}
# {"name": "langchain.workflow", "context": {"trace_id": "0xcb7359e268252dc33578c7cea5849bb9", "span_id": "0x7876cedb29936d57", "trace_state": "[]"}, "kind": "SpanKind.INTERNAL", "parent_id": null, "start_time": "2024-10-22T06:29:22.489473Z", "end_time": "2024-10-22T06:29:25.228174Z", "status": {"status_code": "UNSET"}, "attributes": {"session.session_id": "0x4fa6d91d1f2a4bdbb7a1287d90ec4a16", "workflow_name": "langchain_app_1", "workflow_type": "workflow.langchain"}, "events": [{"name": "data.input", "timestamp": "2024-10-22T06:29:22.489539Z", "attributes": {"input": "What are common ways of doing it?"}}, {"name": "data.output", "timestamp": "2024-10-22T06:29:25.228166Z", "attributes": {"input": "What are common ways of doing it?", "answer": "Task decomposition can be commonly achieved through methods such as using prompting techniques like Chain of Thought or Tree of Thoughts to guide models in breaking down complex tasks into smaller steps. Additionally, task-specific instructions can be provided to direct the agent in performing specific subtasks relevant to the overall goal. Human inputs can also play a crucial role in task decomposition by providing guidance and insights into how tasks can be effectively divided and executed."}}], "links": [], "resource": {"attributes": {"service.name": "langchain_app_1"}, "schema_url": ""}}