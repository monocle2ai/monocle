GEMINI_OUTPUT_PROCESSOR = {
    "type": "inference",
    "attributes": [
        [
            {
                "_comment": "provider type, name, deployment",
                "attribute": "type",
                "accessor": lambda arguments: "google",
            },
            {"attribute": "provider_name", "accessor": lambda arguments: "Google"},
            {
                "attribute": "deployment",
                "accessor": lambda arguments: arguments["instance"]._model_name,
            },
        ],
        [
            {
                "_comment": "LLM Model",
                "attribute": "name",
                "accessor": lambda arguments: arguments["instance"]._model_name,
            },
            {
                "attribute": "type",
                "_comment": "model.llm.<model_name>",
                "accessor": lambda arguments: f"model.llm.{arguments['instance']._model_name}",
            },
        ],
    ],
    "events": [
        {
            "name": "data.input",
            "_comment": "input to Gemini",
            "attributes": [
                {
                    "attribute": "input",
                    "accessor": lambda arguments: (
                        [
                            (
                                arguments["args"][0]
                                if arguments["args"]
                                else arguments["kwargs"].get("contents", "")
                            )
                        ]
                    ),
                }
            ],
        },
        {
            "name": "data.output",
            "_comment": "output from Gemini",
            "attributes": [
                {
                    "attribute": "response",
                    "accessor": lambda arguments: [
                        (
                            arguments["result"].text
                            if hasattr(arguments["result"], "text")
                            else str(arguments["result"])
                        )
                    ],
                }
            ],
        },
        {
            "name": "metadata",
            "attributes": [
                {
                    "_comment": "metadata from Gemini response",
                    "accessor": lambda arguments: {
                        "prompt_tokens": arguments["result"].usage_metadata.prompt_token_count,
                        "completion_tokens": arguments["result"].usage_metadata.candidates_token_count,
                        "total_tokens": arguments["result"].usage_metadata.total_token_count,
                    },
                }
            ],
        },
    ],
}
