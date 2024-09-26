# Monocle Entities
The entity type defines the type of GenAI component that Monocle understand. The monocle instrumentation can extract the relevenat information for this entity. There are a fixed set of [entity types](./entity_types.py) that are defined by Monocle out of the box, eg workflow, model etc. As the GenAI landscape evolves, the Monocle community will introduce a new entity type if the current entities won't represent a new technology component.

## Entity Types
### MonocleEntity.Workflow
Workflow ie the core application code. Supported types are -
- Generic
- Langchain
- LlamaIndex
- Haystack

### MonocleEntity.Model
GenAI models. Supported types are -
- Generic
- LLM
- Embedding

### MonocleEntity.AppHosting
Application host services where the workflow code is run. Supported types are -
- Generic
- AWS_lambda
- AWS_sagemaker
- Azure_func
- Github_codespace
- Azure_mlw

### MonocleEntity.Inference
The model hosting infrastructure services. Supported types are -
- Generic
- NVIDIA_triton
- OpenAI
- Azure_oai
- AWS_sagemaker
- AWS_bedrock
- HuggingFace

### MonocleEntity.VectorStore
Vector search data stores. Supported types are -
- Generic
- Chroma
- AWS_es
- Milvus
- Pinecone
