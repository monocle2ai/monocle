from monocle_apptrace.instrumentation.common.wrapper import atask_wrapper
from monocle_apptrace.instrumentation.metamodel.agentcore.entities.http import AGENTCORE_PROCESSOR

AGENTCORE_METHODS = [
    {
        "package": "bedrock_agentcore.runtime.app",
        "object": "BedrockAgentCoreApp",
        "method": "_invoke_handler",
        "wrapper_method": atask_wrapper,
        "output_processor": AGENTCORE_PROCESSOR
    }
]