# Monocle meta model:
# Monocle Entities --> Entity Type --> Entity

import enum

class MonocleEntity(enum):
    # Supported Workflow/language frameworks
    class Workflow(enum):
        generic = 0
        langchain = 1
        llama_index = 2
        haystack = 3

    # Supported model types
    class Model(enum):
        generic = 0
        llm = 1
        embedding = 2

    # Support Vector databases
    class VectorStore(enum):
        generic = 0
        chroma = 1
        aws_es = 2
        Milvus = 3
        Pinecone = 4

    # Support application hosting frameworks
    class AppHosting(enum):
        generic = 0
        aws_lambda = 1
        aws_sagemaker = 2
        azure_func = 3
        github_codespace = 4
        azure_mlw = 5

    # Supported inference infra/services
    class Inference(enum):
        generic = 0
        nvidia_triton = 1
        openai = 2
        azure_oai = 3
        aws_sagemaker = 4
        aws_bedrock = 5
        hugging_face = 6

class SpanType(enum):
    internal = 0
    retrieval = 2
    inference = 3
    workflow = 4