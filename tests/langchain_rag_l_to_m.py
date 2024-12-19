import os
import uuid
from operator import itemgetter

import bs4
from langchain import hub
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

from monocle_apptrace.instrumentation.common.instrumentor import (
    set_context_properties,
    setup_monocle_telemetry,
)

os.environ["USER_AGENT"] = "langchain-python-app"

setup_monocle_telemetry(
    workflow_name="raanne_rag_ltom",
    span_processors=[BatchSpanProcessor(ConsoleSpanExporter())],
    wrapper_methods=[])

set_context_properties({"session_id": f"{uuid.uuid4().hex}"})

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
print(f"retriever tags:{retriever.tags}")

decompostion_prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful assistant that can break down complex questions into simpler parts. \n
        Your goal is to decompose the given question into multiple sub-questions that can be answerd in isolation to answer the main question in the end. \n
        Provide these sub-questions separated by the newline character. \n
        Original question: {question}\n
        Output (3 queries): 
    """
)

query_generation_chain = (
        {"question": RunnablePassthrough()}
        | decompostion_prompt
        | ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.7)
        | StrOutputParser()
        | (lambda x: x.split("\n"))
)

query = "What is Task Decomposition?"
questions = query_generation_chain.invoke(query)  # What are the benefits of QLoRA

# print(questions)

# Create the final prompt template to answer the question with provided context and background Q&A pairs
template = """Here is the question you need to answer:

\n --- \n {question} \n --- \n

Here is any available background question + answer pairs:

\n --- \n {q_a_pairs} \n --- \n

Here is additional context relevant to the question: 

\n --- \n {context} \n --- \n

Use the above context and any background question + answer pairs to answer the question: \n {question}
"""

least_to_most_prompt = ChatPromptTemplate.from_template(template)
llm = ChatOpenAI(model='gpt-4', temperature=0)

least_to_most_chain = (
        {'context': itemgetter('question') | retriever,
         'q_a_pairs': itemgetter('q_a_pairs'),
         'question': itemgetter('question'),
         }
        | least_to_most_prompt
        | llm
        | StrOutputParser()
)

q_a_pairs = ""
for q in questions:
    answer = least_to_most_chain.invoke({"question": q, "q_a_pairs": q_a_pairs})
    q_a_pairs += f"Question: {q}\n\nAnswer: {answer}\n\n"

# final RAG step
response = least_to_most_chain.invoke({"question": query, "q_a_pairs": q_a_pairs})

print("*************** OUTPUT ********************")
print("\n")
print(f"query: {query}")
print("\n")
print(f"response: {response}")
print("\n")
print(f"q_and_a:\n[{q_a_pairs}]")
print("*************** END ********************")
print(least_to_most_chain)

#*************** OUTPUT ********************


# query: What is Task Decomposition?
#
#
# response: Task Decomposition is the process of breaking down a complex task into smaller, more manageable subtasks or steps. This technique is often used in artificial intelligence systems to enhance model performance on complex tasks. The model is instructed to "think step by step" to utilize more computation to decompose hard tasks into simpler steps. This process transforms big tasks into multiple manageable tasks and provides insight into the model's thinking process. Task decomposition can be done by using simple prompting, task-specific instructions, or with human inputs. It also allows for the exploration of multiple reasoning possibilities at each step, as seen in the Tree of Thoughts method. This can lead to a more thorough and comprehensive approach to problem-solving. Task decomposition is a crucial technique for making tasks more manageable and improving the performance of AI models.
#
#
# q_and_a:
# [Question: - What is the definition of task decomposition?
#
# Answer: Task decomposition is the process of breaking down a complex task into smaller, more manageable steps or subtasks. This technique is often used in artificial intelligence systems to enhance model performance on complex tasks. The model is instructed to "think step by step" to utilize more computation to decompose hard tasks into simpler steps. This process transforms big tasks into multiple manageable tasks and provides insight into the model's thinking process. Task decomposition can be done by using simple prompting, task-specific instructions, or with human inputs.
#
# Question: - Why is task decomposition important?
#
# Answer: Task decomposition is important for several reasons. Firstly, it breaks down complex tasks into smaller, more manageable steps or subtasks, making it easier to understand and execute the task. This is particularly useful in artificial intelligence systems, where it can enhance model performance on complex tasks. By instructing the model to "think step by step", it can utilize more computation to simplify hard tasks.
#
# Secondly, task decomposition transforms big tasks into multiple manageable tasks, providing insight into the model's thinking process. This can be beneficial for understanding how the model is approaching the task and for troubleshooting any issues that may arise.
#
# Thirdly, task decomposition allows for the exploration of multiple reasoning possibilities at each step, as seen in the Tree of Thoughts method. This can lead to a more thorough and comprehensive approach to problem-solving.
#
# Lastly, task decomposition can be a useful tool for long-term planning and exploring the solution space, although this remains a challenging area in artificial intelligence. Despite these challenges, task decomposition is a crucial technique for making tasks more manageable and improving the performance of AI models.
#
# Question: - What are the steps involved in task decomposition?
#
# Answer: Task decomposition involves the following steps:
#
# 1. Identification of the Complex Task: The first step in task decomposition is to identify the complex task that needs to be broken down. This could be anything from writing a novel to solving a complex mathematical problem.
#
# 2. Breaking Down the Task: The next step is to break down the complex task into smaller, more manageable subtasks. This can be done by using simple prompting, such as "Steps for XYZ.\\n1.", or by asking questions like "What are the subgoals for achieving XYZ?".
#
# 3. Use of Task-Specific Instructions: Task-specific instructions can also be used to decompose the task. For example, if the task is to write a novel, the instruction could be "Write a story outline."
#
# 4. Exploration of Multiple Reasoning Possibilities: The Tree of Thoughts method can be used to explore multiple reasoning possibilities at each step. This involves decomposing the problem into multiple thought steps and generating multiple thoughts per step, creating a tree structure.
#
# 5. Evaluation: Each state or step can be evaluated by a classifier (via a prompt) or majority vote. This helps to determine the most effective approach to the task.
#
# 6. Execution: Once the task has been decomposed and the best approach determined, the subtasks can be executed. This could involve delegating tasks to AI models or completing them manually.
#
# 7. Review and Analysis: The final step in task decomposition is to review and analyze the process and results. This can help to refine the approach for future tasks.
#
# ]
# *************** END ********************
# first={
#   context: RunnableLambda(itemgetter('question'))
#            | VectorStoreRetriever(tags=['Chroma', 'OpenAIEmbeddings'], vectorstore=<langchain_chroma.vectorstores.Chroma object at 0x71864ab6c8b0>),
#   q_a_pairs: RunnableLambda(itemgetter('q_a_pairs')),
#   question: RunnableLambda(itemgetter('question'))
# }
# middle=[ChatPromptTemplate(input_variables=['context', 'q_a_pairs', 'question'], messages=[HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['context', 'q_a_pairs', 'question'], template='Here is the question you need to answer:\n\n\n --- \n {question} \n --- \n\n\nHere is any available background question + answer pairs:\n\n\n --- \n {q_a_pairs} \n --- \n\n\nHere is additional context relevant to the question: \n\n\n --- \n {context} \n --- \n\n\nUse the above context and any background question + answer pairs to answer the question: \n {question}\n'))]), ChatOpenAI(client=<openai.resources.chat.completions.Completions object at 0x71864a9d51b0>, async_client=<openai.resources.chat.completions.AsyncCompletions object at 0x71864a9d69b0>, model_name='gpt-4', temperature=0.0, openai_api_key=SecretStr('**********'), openai_proxy='')]
# last=StrOutputParser()
