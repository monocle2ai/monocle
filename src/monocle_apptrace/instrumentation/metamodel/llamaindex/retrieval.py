from monocle_apptrace.instrumentation.metamodel.llamaindex._helper import (
  extract_vectorstore_deployment,
)

RETRIEVAL = {
  "type": "retrieval",
  "attributes": [
    [
      {
        "_comment": "vector store name and type",
        "attribute": "name",
        "accessor": lambda arguments: type(arguments['instance']._vector_store).__name__
      },
      {
        "attribute": "type",
        "accessor": lambda arguments: 'vectorstore.'+type(arguments['instance']._vector_store).__name__
      },
      {
        "attribute": "deployment",
        "accessor": lambda arguments: extract_vectorstore_deployment(arguments['instance']._vector_store)
      }
    ],
    [
      {
        "_comment": "embedding model name and type",
        "attribute": "name",
        "accessor": lambda arguments: arguments['instance']._embed_model.model_name
      },
      {
        "attribute": "type",
        "accessor": lambda arguments: 'model.embedding.'+arguments['instance']._embed_model.model_name
      }
    ]
  ]
}
