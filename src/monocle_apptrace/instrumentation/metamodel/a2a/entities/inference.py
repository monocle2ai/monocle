from monocle_apptrace.instrumentation.metamodel.a2a import _helper

A2A_CLIENT = {
    "type": "agentic.invocation",
    "attributes": [
        [
            {
                "attribute": "type",
                "accessor": lambda arguments: "agent2agent.server"
            },
            {
                "attribute": "url",
                "accessor": lambda arguments: _helper.get_url(arguments)
            },
            {
                "attribute": "method",
                "accessor": lambda arguments: _helper.get_method(arguments)
            }
        ]
    ],
    "events": [
        {
            "name": "data.input",
            "attributes": [
                {
                    "_comment": "this is a2a input",
                    "attribute": "input",
                    "accessor": lambda arguments: _helper.get_params_arguments(arguments)
                },
                {
                    "_comment": "this is a2a input",
                    "attribute": "role",
                    "accessor": lambda arguments: _helper.get_role(arguments)
                },
            ],
        },
        {
            "name": "data.output",
            "attributes": [
                {
                    "_comment": "this is a2a output",
                    "attribute": "status",
                    "accessor": lambda arguments: _helper.get_status(arguments, "status")
                },
                {
                    "_comment": "this is a2a output",
                    "attribute": "response",
                    "accessor": lambda arguments: _helper.get_response(arguments)   
                },
            ],
        },
    ],
}

# A2A_RESOLVE = {
#     "type": "a2a.resolve",
#     "attributes": [
#         [
#             # {
#             #   "_comment": "tool type",
#             #   "attribute": "type",
#             #   "accessor": lambda arguments:'tool.mcp'
#             # },
#             {
#                 "_comment": "name of the tool",
#                 "attribute": "name",
#                 "accessor": lambda arguments: _helper.log(arguments),
#             },
#                         {
#                 "_comment": "tool description",
#                 "attribute": "agent_description",
#                 "accessor": lambda arguments: arguments["result"].description
#             },
#             {
#                 "_comment": "tool name",
#                 "attribute": "agent_name",
#                 "accessor": lambda arguments: arguments["result"].name
#             }
#             # {
#             #     "_comment": "tool type",
#             #     "attribute": "type",
#             #     "accessor": lambda arguments: _helper.get_type(arguments),
#             # },
#         ]
#     ],
#     "events": [
#         # {
#         #     "name": "data.input",
#         #     "attributes": [
#         #         {
#         #             "_comment": "this is Tool input",
#         #             "attribute": "input",
#         #             "accessor": lambda arguments: _helper.get_params_arguments(
#         #                 arguments
#         #             ),
#         #         },
#         #     ],
#         # },
#         # {
#         #     "name": "data.output",
#         #     "attributes": [
#         #         {
#         #             "_comment": "this is Tool output",
#         #             "attribute": "output",
#         #             "accessor": lambda arguments: _helper.get_output_text(arguments)
#         #         },
#         #     ],
#         # },
#     ],
# }


