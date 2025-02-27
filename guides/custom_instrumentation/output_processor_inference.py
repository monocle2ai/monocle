INFERENCE_OUTPUT_PROCESSOR = {
    "type": "inference",
    "attributes": [
        [
            {
                "_comment": "provider type ,name , deployment , inference_endpoint",
                "attribute": "type",
                "accessor": lambda arguments: "openai"
            },
            {
                "attribute": "provider_name",
                "accessor": lambda arguments: "OpenAI"
            },
            {
                "attribute": "deployment",
                "accessor": lambda arguments: arguments['kwargs'].get('model', 'unknown')
            },
            {
                "attribute": "inference_endpoint",
                "accessor": lambda arguments: arguments['instance'].base_url
            }
        ],
        [
            {
                "_comment": "LLM Model",
                "attribute": "name",
                "accessor": lambda arguments: arguments['kwargs'].get('model', 'unknown')
            },
            {
                "attribute": "type",
                "_comment": "model.llm.<model_name>",
                "accessor": lambda arguments: f"model.llm.{arguments['kwargs'].get('model', 'unknown')}"
            }
        ]
    ],
    "events": [
        {"name": "data.input",
         "_comment": "",
         "attributes": [
             {
                 "_comment": "this is input to LLM, the accessor extracts only the message contents",
                 "attribute": "input",
                 "accessor": lambda arguments: [
                     msg["content"] 
                     for msg in arguments['kwargs'].get('messages', [])
                 ] if isinstance(arguments['kwargs'].get('messages'), list) else []
             }
         ]
         },
        {
            "name": "data.output",
            "_comment": "",
            "attributes": [
                {
                    "_comment": "this is output from LLM, it includes the string response which is part of a list",
                    "attribute": "response",
                    "accessor": lambda arguments: arguments['result']['choices'][0]['message']['content'] if 'result' in arguments and 'choices' in arguments['result'] else None
                }
            ]
        },
        {
            "name": "metadata",
            "attributes": [
                {
                    "_comment": "this is metadata usage from LLM",
                    "accessor": lambda arguments: arguments['result']['usage'] if 'result' in arguments and 'usage' in arguments['result'] else {}
                }
            ]
        }
    ]
}
