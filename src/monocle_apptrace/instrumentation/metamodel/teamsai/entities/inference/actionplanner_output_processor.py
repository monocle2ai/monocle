from monocle_apptrace.instrumentation.metamodel.teamsai import (
    _helper,
)
ACTIONPLANNER_OUTPUT_PROCESSOR = {
    "type": "generic",
    "attributes": [
        [
            {
                "_comment": "planner type and configuration",
                "attribute": "type",
                "accessor": lambda arguments: "teams.planner"
            },
            {
                "attribute": "planner_type",
                "accessor": lambda arguments: "ActionPlanner"
            },
            {
                "attribute": "max_repair_attempts",
                "accessor": lambda arguments: arguments["instance"]._options.max_repair_attempts if hasattr(arguments["instance"], "_options") else 3
            }
        ],
        [
            {
                "_comment": "model configuration",
                "attribute": "model",
                "accessor": lambda arguments: arguments["instance"]._options.model.__class__.__name__ if hasattr(arguments["instance"], "_options") else "unknown"
            },
            {
                "attribute": "tokenizer",
                "accessor": lambda arguments: arguments["instance"]._options.tokenizer.__class__.__name__ if hasattr(arguments["instance"], "_options") else "GPTTokenizer"
            }
        ]
    ],
    "events": [
        {
            "name": "data.input",
            "_comment": "input configuration to ActionPlanner",
            "attributes": [
               {
                    "_comment": "execution metadata",
                    "accessor": lambda arguments: _helper.get_prompt_template(arguments)
                },
                {
                    "attribute": "validator",
                    "accessor": lambda arguments: arguments["kwargs"].get("validator").__class__.__name__ if arguments.get("kwargs", {}).get("validator") else "DefaultResponseValidator"
                },
                {
                    "attribute": "memory_type",
                    "accessor": lambda arguments: arguments["kwargs"].get("memory").__class__.__name__ if arguments.get("kwargs", {}).get("memory") else "unknown"
                }
            ]
        },
        {
            "name": "data.output",
            "_comment": "output from ActionPlanner",
            "attributes": [
                {
                    "attribute": "status",
                    "accessor": lambda arguments: _helper.status_check(arguments)
                },
                {
                    "attribute": "response",
                    "accessor": lambda arguments: arguments["result"].message.content if hasattr(arguments["result"], "message") else str(arguments["result"])
                }
            ]
        },
        {
            "name": "metadata",
            "attributes": [
                {
                    "_comment": "execution metadata",
                    "accessor": lambda arguments: {
                        "latency_ms": arguments.get("latency_ms"),
                        "feedback_enabled": arguments["instance"]._enable_feedback_loop if hasattr(arguments["instance"], "_enable_feedback_loop") else False
                    }
                }
            ]
        }
    ]
}