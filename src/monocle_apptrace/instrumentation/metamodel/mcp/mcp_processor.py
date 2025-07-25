from opentelemetry.context import set_value, attach, detach
from monocle_apptrace.instrumentation.common.span_handler import SpanHandler
from monocle_apptrace.instrumentation.metamodel.mcp._helper import (
   get_name

)
from monocle_apptrace.instrumentation.metamodel.langgraph.entities.inference import (
    AGENT_DELEGATION, AGENT_REQUEST
)

class MCPAgentHandler(SpanHandler):
    def skip_span(self, to_wrap, wrapped, instance, args, kwargs) -> bool:
        return get_name({"args": args, "kwargs": kwargs}) is None or args[0].root.method == "tools/list"
