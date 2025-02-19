from monocle_apptrace.instrumentation.common.wrapper import task_wrapper
from monocle_apptrace.instrumentation.metamodel.langgraph.entities.inference import (
    INFERENCE,
)
LANGGRAPH_METHODS = [
    {
        "package": "langgraph.graph.state",
         "object": "CompiledStateGraph",
         "method": "invoke",
         "span_name": "langgraph.graph.invoke",
         "wrapper_method": task_wrapper,
         "output_processor": INFERENCE
    }
]