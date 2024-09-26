# Monocle meta model:
# Monocle Entities --> Entity Type --> Entity

import enum

class MonocleEntity(enum):
    # Supported Workflow/language frameworks
    class Workflow(enum):
        Generic = 0
        Langchain = 1
        LlamaIndex = 2
        Haystack = 3

    # Supported model types
    class Model(enum):
        Generic = 0
        LLM = 1
        Embedding = 2

    # Support Vector databases
    class VectorStore(enum):
        Generic = 0
        Chroma = 1
        AWS_es = 2
        Milvus = 3
        Pinecone = 4

    # Support application hosting frameworks
    class AppHosting(enum):
        Generic = 0
        AWS_lambda = 1
        AWS_sagemaker = 2
        Azure_func = 3
        Github_codespace = 4
        Azure_mlw = 5

    # Supported inference infra/services
    class Inference(enum):
        Generic = 0
        NVIDIA_triton = 1
        OpenAI = 2
        Azure_oai = 3
        AWS_sagemaker = 4
        AWS_bedrock = 5
        HuggingFace = 6

class SpanType(enum):
    Internal = 0
    Retrieval = 2
    Inference = 3
    Workflow = 4