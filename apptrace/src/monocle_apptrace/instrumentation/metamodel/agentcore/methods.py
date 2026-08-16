from monocle_apptrace.instrumentation.common.wrapper import atask_wrapper
from monocle_apptrace.instrumentation.metamodel.agentcore.entities.http import AGENTCORE_PROCESSOR

AGENTCORE_METHODS = [
    {
        "package": "bedrock_agentcore.runtime.app",
        "object": "BedrockAgentCoreApp",
        "method": "_invoke_handler",
        "wrapper_method": atask_wrapper,
        "output_processor": AGENTCORE_PROCESSOR
    },
    {
        # Wraps the caller of _invoke_handler, which is where the response is
        # built: the scope tagging this invocation's spans has to be active
        # while the agent runs, and the trailer can only be appended once the
        # result has been serialized into a Response. Carries no span of its
        # own — _invoke_handler above already reports the invocation.
        "package": "bedrock_agentcore.runtime.app",
        "object": "BedrockAgentCoreApp",
        "method": "_handle_invocation",
        "wrapper_method": atask_wrapper,
        "span_handler": "agentcore_handler",
        "skip_span": True
    }
]