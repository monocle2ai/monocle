
VECTOR_OUTPUT_PROCESSOR = {
    "type": "retrieval",
    "attributes": [
        [
            {
                "_comment": "vector store name and type",
                "attribute": "name",
                "accessor": lambda arguments: type(arguments["instance"]).__name__,
            },
            {
                "attribute": "type",
                "accessor": lambda arguments: "vectorstore."
                + type(arguments["instance"]).__name__,
            },
            {
                "attribute": "deployment",
                "accessor": lambda arguments: ""
            },
        ],
        [
            {
                "_comment": "embedding model name and type",
                "attribute": "name",
                "accessor": lambda arguments: arguments["instance"].model
            },
            {
                "attribute": "type",
                "accessor": lambda arguments: "model.embedding." + arguments["instance"].model
            },
        ],
    ],
    "events": [
        {
            "name": "data.input",
            "_comment": "query input to vector store",
            "attributes": [
                {
                    "attribute": "input",
                    "accessor": lambda arguments: arguments["args"][0] if arguments["args"] else None
                }
            ],
        },
        {
            "name": "data.output",
            "_comment": "results from vector store search",
            "attributes": [
                {
                    "attribute": "response",
                    "accessor": lambda arguments: ", ".join([resultItem['metadata'].get('text', '') for resultItem in arguments["result"]])

                }
            ],
        }
    ],
}
