import os
import bs4
from langsmith import Client
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
# from langchain_mistralai import ChatMistralAI
from langchain_openai import (
    AzureChatOpenAI,
    AzureOpenAI,
    ChatOpenAI,
    OpenAI,
    OpenAIEmbeddings,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate


class TestScopes:
    def config_scope_func(self, message):
        chain = setup_chain()        
        result = chain.invoke(message)
        return result

    async def config_scope_async_func(self, message):
        chain = setup_chain()
        result = chain.invoke(message)
        return result

@staticmethod
def setup_chain():
    """ Setup the langchain chain for the test """
    llm = AzureChatOpenAI(
        # engine=os.environ.get("AZURE_OPENAI_API_DEPLOYMENT"),
        azure_deployment=os.environ.get("AZURE_OPENAI_API_DEPLOYMENT"),
        api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        temperature=0.1,
        # model="gpt-4",

        model="gpt-3.5-turbo-0125")
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
    return rag_chain

def setup_simple_chain():
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate([
        ("system", "You are an assistant for question-answering tasks."
            + "If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise."
            + "Respond in the same sentiment as the question"),
        ("user", "Question: {question}")
    ])

    simple_chain = (
        {"question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return simple_chain

def exec_chain(message):
    chain = setup_chain()
    return chain.invoke(message)
