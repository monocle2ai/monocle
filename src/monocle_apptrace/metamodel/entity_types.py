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
    class VectorDB(enum):
        Generic = 0
        Milvus = 1
        Chroma = 2
        Vespa = 3
        LanceDB = 4
        Pinecone = 5
        Viviate = 6
        PostgreS = 7
        Openstore = 8
        Clickhouse = 9
        ElasticSearch = 10
        SingleStore = 11
        CouchDB = 12

    # Support application hosting frameworks
    class AppHosting(enum):
        Generic = 0
        AWS_Lambda = 1
        AWS_Sagemaker = 2
        Azure_Function = 3
        Github_Codespace = 4
        Azure_ML = 5

    # Supported inference infra/services
    class Inference(enum):
        Generic = 0
        NVIDIA_Triton = 1
        OpenAI = 2
        Azure_OpenAI = 3
        AWS_Sagemaker = 4
        AWS_Bedrock = 5
        HuggingFace = 6
        Cohere = 7
        vLLM = 8
