# pylint: disable=too-few-public-methods
from typing import Any, Dict
from monocle_apptrace.instrumentation.common.wrapper import task_wrapper, scope_wrapper
from monocle_apptrace.instrumentation.common.span_handler import SpanHandler, NonFrameworkSpanHandler
from monocle_apptrace.instrumentation.metamodel.azureaiinference.methods import AZURE_AI_INFERENCE_METHODS
from monocle_apptrace.instrumentation.metamodel.botocore.methods import BOTOCORE_METHODS
from monocle_apptrace.instrumentation.metamodel.botocore.handlers.botocore_span_handler import BotoCoreSpanHandler
from monocle_apptrace.instrumentation.metamodel.hugging_face.methods import HUGGING_FACE_METHODS
from monocle_apptrace.instrumentation.metamodel.langchain.methods import (
    LANGCHAIN_METHODS,
)
from monocle_apptrace.instrumentation.metamodel.llamaindex.methods import (LLAMAINDEX_METHODS, )
from monocle_apptrace.instrumentation.metamodel.llamaindex.llamaindex_processor import LlamaIndexToolHandler, LlamaIndexAgentHandler, LlamaIndexSingleAgenttToolHandlerWrapper
from monocle_apptrace.instrumentation.metamodel.haystack.methods import (HAYSTACK_METHODS, )
from monocle_apptrace.instrumentation.metamodel.openai.methods import (OPENAI_METHODS,)
from monocle_apptrace.instrumentation.metamodel.openai.openai_processor import ( OpenAISpanHandler, OpenAIAgentsSpanHandler)
from monocle_apptrace.instrumentation.metamodel.langgraph.methods import LANGGRAPH_METHODS
from monocle_apptrace.instrumentation.metamodel.langgraph.langgraph_processor import LanggraphAgentHandler, LanggraphToolHandler
from monocle_apptrace.instrumentation.metamodel.agents.methods import AGENTS_METHODS
from monocle_apptrace.instrumentation.metamodel.agents.agents_processor import AgentsSpanHandler
from monocle_apptrace.instrumentation.metamodel.flask.methods import (FLASK_METHODS, )
from monocle_apptrace.instrumentation.metamodel.flask._helper import FlaskSpanHandler, FlaskResponseSpanHandler
from monocle_apptrace.instrumentation.metamodel.requests.methods import (REQUESTS_METHODS, )
from monocle_apptrace.instrumentation.metamodel.requests._helper import RequestSpanHandler
from monocle_apptrace.instrumentation.metamodel.teamsai.methods import (TEAMAI_METHODS, )
from monocle_apptrace.instrumentation.metamodel.anthropic.methods import (ANTHROPIC_METHODS, )
from monocle_apptrace.instrumentation.metamodel.aiohttp.methods import (AIOHTTP_METHODS, )
from monocle_apptrace.instrumentation.metamodel.aiohttp._helper import aiohttpSpanHandler
from monocle_apptrace.instrumentation.metamodel.azfunc._helper import (azureSpanHandler)
from monocle_apptrace.instrumentation.metamodel.azfunc.methods import AZFUNC_HTTP_METHODS
from monocle_apptrace.instrumentation.metamodel.gemini.methods import GEMINI_METHODS
from monocle_apptrace.instrumentation.metamodel.fastapi.methods import FASTAPI_METHODS
from monocle_apptrace.instrumentation.metamodel.fastapi._helper import FastAPISpanHandler, FastAPIResponseSpanHandler
from monocle_apptrace.instrumentation.metamodel.fastmcp.methods import FASTMCP_METHODS
from monocle_apptrace.instrumentation.metamodel.lambdafunc._helper import lambdaSpanHandler
from monocle_apptrace.instrumentation.metamodel.lambdafunc.methods import LAMBDA_HTTP_METHODS
from monocle_apptrace.instrumentation.metamodel.mcp.methods import MCP_METHODS
from monocle_apptrace.instrumentation.metamodel.mcp.mcp_processor import MCPAgentHandler
from monocle_apptrace.instrumentation.metamodel.a2a.methods import A2A_CLIENT_METHODS
from monocle_apptrace.instrumentation.metamodel.litellm.methods import LITELLM_METHODS
from monocle_apptrace.instrumentation.metamodel.adk.methods import ADK_METHODS
from monocle_apptrace.instrumentation.metamodel.mistral.methods import MISTRAL_METHODS
from monocle_apptrace.instrumentation.metamodel.strands.methods import STRAND_METHODS
from monocle_apptrace.instrumentation.metamodel.strands.strands_processor import StrandsSpanHandler
from monocle_apptrace.instrumentation.metamodel.adk._helper import AdkSpanHandler

class WrapperMethod:
    def __init__(
            self,
            package: str,
            object_name: str,
            method: str,
            span_name: str = None,
            output_processor : str = None,
            wrapper_method = task_wrapper,
            span_handler = 'default',
            scope_name: str = None,
            span_type: str = None,
            scope_values = None,
            ):
        self.package = package
        self.object = object_name
        self.method = method
        self.span_name = span_name
        self.output_processor=output_processor
        self.span_type = span_type
        self.scope_values = scope_values

        self.span_handler:SpanHandler.__class__ = span_handler
        self.scope_name = scope_name
        if scope_name and not scope_values:
            self.wrapper_method = scope_wrapper
        else:
            self.wrapper_method = wrapper_method

    def to_dict(self) -> dict:
        # Create a dictionary representation of the instance
        instance_dict = {
            'package': self.package,
            'object': self.object,
            'method': self.method,
            'span_name': self.span_name,
            'output_processor': self.output_processor,
            'wrapper_method': self.wrapper_method,
            'span_handler': self.span_handler,
            'scope_name': self.scope_name,
            'span_type': self.span_type,
            'scope_values': self.scope_values,
        }
        return instance_dict

    def get_span_handler(self) -> SpanHandler:
        return self.span_handler()

DEFAULT_METHODS_LIST = (
    LANGCHAIN_METHODS + 
    LLAMAINDEX_METHODS + 
    HAYSTACK_METHODS + 
    BOTOCORE_METHODS + 
    FLASK_METHODS + 
    REQUESTS_METHODS + 
    LANGGRAPH_METHODS + 
    AGENTS_METHODS +
    OPENAI_METHODS + 
    TEAMAI_METHODS +
    ANTHROPIC_METHODS + 
    AIOHTTP_METHODS + 
    AZURE_AI_INFERENCE_METHODS + 
    AZFUNC_HTTP_METHODS + 
    GEMINI_METHODS + 
    FASTAPI_METHODS + 
    FASTMCP_METHODS +
    LAMBDA_HTTP_METHODS +
    MCP_METHODS + 
    A2A_CLIENT_METHODS +
    LITELLM_METHODS +
    ADK_METHODS +
    MISTRAL_METHODS +
    HUGGING_FACE_METHODS +
    STRAND_METHODS
)

MONOCLE_SPAN_HANDLERS: Dict[str, SpanHandler] = {
    "default": SpanHandler(),
    "aiohttp_handler": aiohttpSpanHandler(),
    "botocore_handler": BotoCoreSpanHandler(),
    "flask_handler": FlaskSpanHandler(),
    "flask_response_handler": FlaskResponseSpanHandler(),
    "request_handler": RequestSpanHandler(),
    "non_framework_handler": NonFrameworkSpanHandler(),
    "openai_handler": OpenAISpanHandler(),
    "openai_agents_handler": OpenAIAgentsSpanHandler(),
    "azure_func_handler": azureSpanHandler(),
    "mcp_agent_handler": MCPAgentHandler(),
    "fastapi_handler": FastAPISpanHandler(),
    "fastapi_response_handler": FastAPIResponseSpanHandler(),
    "langgraph_agent_handler": LanggraphAgentHandler(),
    "langgraph_tool_handler": LanggraphToolHandler(),
    "agents_agent_handler": AgentsSpanHandler(),
    "llamaindex_tool_handler": LlamaIndexToolHandler(),
    "llamaindex_agent_handler": LlamaIndexAgentHandler(),
    "llamaindex_single_agent_tool_handler": LlamaIndexSingleAgenttToolHandlerWrapper(),
    "lambda_func_handler": lambdaSpanHandler(),
    "adk_handler": AdkSpanHandler(),
    "strands_handler": StrandsSpanHandler()
}
