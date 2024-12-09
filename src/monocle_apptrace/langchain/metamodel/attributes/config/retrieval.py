from monocle_apptrace.utils import get_vectorstore_deployment

retrieval={
  "type": "retrieval",
  "attributes": [
    [
      {
        "_comment": "vector store name and type",
        "attribute": "name",
        "accessor": lambda arguments: type(arguments['instance'].vectorstore).__name__
      },
      {
        "attribute": "type",
        "accessor": lambda arguments: 'vectorstore.'+type(arguments['instance'].vectorstore).__name__
      },
      {
        "attribute": "deployment",
        "accessor": lambda arguments: get_vectorstore_deployment(arguments['instance'].vectorstore.__dict__)
      }
    ],
    [
      {
        "_comment": "embedding model name and type",
        "attribute": "name",
        "accessor": lambda arguments: arguments['instance'].vectorstore.embeddings.model
      },
      {
        "attribute": "type",
        "accessor": lambda arguments: 'model.embedding.'+arguments['instance'].vectorstore.embeddings.model
      }
    ]
  ]
}
