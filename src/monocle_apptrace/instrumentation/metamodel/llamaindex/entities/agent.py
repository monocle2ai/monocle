from monocle_apptrace.instrumentation.metamodel.llamaindex import (
    _helper,
)

AGENT = {
    "type": "agent",
    "attributes": [
        [
            {
                "_comment": "Agent name, type and Tools.",
                "attribute": "name",
                "accessor": lambda arguments: arguments['instance'].__class__.__name__
            },
            {
                "attribute": "type",
                "accessor": lambda arguments: 'Agent.oai'
            },
            {
                "attribute": "tools",
                "accessor": lambda arguments: _helper.extract_tools(arguments['instance'])
            }
        ]

    ],
    "events": [
        {"name": "data.input",
         "attributes": [

             {
                 "_comment": "this is instruction and user query to LLM",
                 "attribute": "input",
                 "accessor": lambda arguments: _helper.extract_messages(arguments['args'])
             }
         ]
         },
        {
            "name": "data.output",
            "attributes": [
                {
                    "_comment": "this is response from LLM",
                    "attribute": "response",
                    "accessor": lambda arguments: _helper.extract_assistant_message(arguments['result'])
                }
            ]
        }
    ]
}
