import json
from monocle_apptrace.instrumentation.metamodel.teamsai import (
    _helper,
)

def get_context(arguments):
    """
    Extracts the context type from the arguments.
    """
    if arguments.get("args") and len(arguments["args"]) > 0 and hasattr(arguments["args"][0], "data") and hasattr(arguments["args"][0].data, "response") and hasattr(arguments["args"][0].data.response, "context"):
        return arguments.get("args")[0].data.response.content

def get_action(arguments):
    """
    Extracts the action from the arguments.
    """
    if arguments.get("args") and len(arguments.get("args")) > 0 and hasattr(arguments.get("args")[0], "data") and hasattr(arguments.get("args")[0].data, "action"):
        return arguments.get("args")[0].data.action
    return None

def get_action_parameters(arguments):
    """
    Extracts the action parameters from the arguments.
    """
    if arguments.get("args") and len(arguments.get("args")) > 0 and hasattr(arguments.get("args")[0], "data") and hasattr(arguments.get("args")[0].data, "parameters"):
        # parameters is dict[str, str], so serialize it to {"key": "value"}
        if isinstance(arguments.get("args")[0].data.parameters, dict):
            return json.dumps(arguments.get("args")[0].data.parameters)
    return None

def get_action_output(arguments):
    """
    Extracts the action output from the arguments.
    """
    # Check if the arguments contain result and it is string
    if arguments.get("result") and isinstance(arguments.get("result"), str):
        return arguments.get("result")
    return None

def get_command_type(arguments):
    """
    Extracts the command type from the arguments.
    """
    if get_action(arguments):
        return "do"
    return "say"

AI_APP_OUTPUT_PROCESSOR = {
    "type": "command",
    "attributes": [
        [
            {
                "attribute": "type",
                "accessor": lambda arguments: "command"
            },
            {
                "attribute": "name",
                "accessor": lambda arguments: get_command_type(arguments)
            },
        ]
        
    ],
    "events": [
        {
            "name": "data.input",
            "_comment": "input configuration to ActionPlanner",
            "attributes": [
                {
                    "attribute": "context",
                    "accessor": lambda arguments: get_context(arguments)
                },
                {
                    "attribute": "action_name",
                    "accessor": lambda arguments: get_action(arguments)
                },
                {
                    "attribute": "action_parameters",
                    "accessor": lambda arguments: get_action_parameters(arguments)
                }
            ]
        },
        {
            "name": "data.output",
            "_comment": "output from ActionPlanner",
            "attributes": [
                {
                    "attribute": "output",
                    "accessor": lambda arguments: get_action_output(arguments)
                }
            ]
        },
        {
            "name": "metadata",
            "attributes": [
                
            ]
        }
    ]
}