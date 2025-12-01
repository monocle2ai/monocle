from monocle_apptrace.instrumentation.common.wrapper import task_wrapper, atask_wrapper, atask_iter_wrapper
from monocle_apptrace.instrumentation.metamodel.strands.entities.agent import AGENT, AGENT_REQUEST
from monocle_apptrace.instrumentation.metamodel.strands.entities.tool import TOOL

STRAND_METHODS = [
    {
        "package": "strands.agent.agent",
        "object": "Agent",
        "method": "__call__",
        "wrapper_method": task_wrapper,
        "span_handler": "strands_handler",
        "output_processor_list": [AGENT_REQUEST, AGENT]
    },
    {
        "package": "strands.tools.executors.concurrent",
        "object": "ConcurrentToolExecutor",
        "method": "_execute",
        "wrapper_method": atask_iter_wrapper,
        "output_processor": TOOL,
    }
]