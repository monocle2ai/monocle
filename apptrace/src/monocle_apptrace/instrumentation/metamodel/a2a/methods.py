from monocle_apptrace.instrumentation.common.wrapper import task_wrapper, atask_wrapper
from monocle_apptrace.instrumentation.metamodel.a2a.entities.inference import A2A_CLIENT

A2A_CLIENT_METHODS = [
    # {
    #     "package": "a2a.client.client",
    #     "object": "A2ACardResolver",
    #     "method": "get_agent_card",
    #     "wrapper_method": atask_wrapper,
    #     # "span_handler": "mcp_agent_handler",
    #     "output_processor": A2A_RESOLVE,
    # },
    {
        "package": "a2a.client",
        "object": "A2AClient",
        "method": "send_message",
        "wrapper_method": atask_wrapper,
        "output_processor": A2A_CLIENT,
    },
]


