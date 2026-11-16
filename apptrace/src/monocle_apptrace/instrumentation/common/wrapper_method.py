# pylint: disable=too-few-public-methods
import logging
from difflib import get_close_matches
from typing import Any, Dict, List, Optional, Union
from monocle_apptrace.instrumentation.common.wrapper import task_wrapper, scope_wrapper
from monocle_apptrace.instrumentation.common.span_handler import SpanHandler, NonFrameworkSpanHandler
from monocle_apptrace.instrumentation.metamodel.azureaiinference.methods import AZURE_AI_INFERENCE_METHODS
from monocle_apptrace.instrumentation.metamodel.botocore.methods import BOTOCORE_METHODS
from monocle_apptrace.instrumentation.metamodel.botocore.handlers.botocore_span_handler import BotoCoreSpanHandler
from monocle_apptrace.instrumentation.metamodel.claude_cli.claude_span_handler import ClaudeSpanHandler
from monocle_apptrace.instrumentation.metamodel.hugging_face.methods import HUGGING_FACE_METHODS
from monocle_apptrace.instrumentation.metamodel.langchain.methods import (
    LANGCHAIN_METHODS,
)
from monocle_apptrace.instrumentation.metamodel.llamaindex.methods import (LLAMAINDEX_METHODS, )
from monocle_apptrace.instrumentation.metamodel.llamaindex.llamaindex_processor import LlamaIndexToolHandler, LlamaIndexAgentHandler, LlamaIndexSingleAgenttToolHandlerWrapper, LlamaIndexWorkflowHandler
from monocle_apptrace.instrumentation.metamodel.haystack.methods import (HAYSTACK_METHODS, )
from monocle_apptrace.instrumentation.metamodel.openai.methods import (OPENAI_METHODS,)
from monocle_apptrace.instrumentation.metamodel.openai.openai_processor import ( OpenAISpanHandler, OpenAIAgentsSpanHandler)
from monocle_apptrace.instrumentation.metamodel.langgraph.methods import LANGGRAPH_METHODS
from monocle_apptrace.instrumentation.metamodel.langgraph.langgraph_processor import LanggraphAgentHandler, LanggraphToolHandler
from monocle_apptrace.instrumentation.metamodel.crew_ai.methods import CREW_AI_METHODS
from monocle_apptrace.instrumentation.metamodel.crew_ai.crew_ai_processor import CrewAIAgentHandler, CrewAITaskHandler, CrewAIToolHandler
from monocle_apptrace.instrumentation.metamodel.msagent.methods import MSAGENT_METHODS
from monocle_apptrace.instrumentation.metamodel.msagent.msagent_processor import MSAgentRequestHandler, MSAgentAgentHandler, MSAgentInferenceHandler, MSAgentInferenceStreamHandler, MSAgentToolHandler, MSAgentOrchestratorHandler
from monocle_apptrace.instrumentation.metamodel.agents.methods import AGENTS_METHODS
from monocle_apptrace.instrumentation.metamodel.agents.agents_processor import AgentsSpanHandler
from monocle_apptrace.instrumentation.metamodel.flask.methods import (FLASK_METHODS, )
from monocle_apptrace.instrumentation.metamodel.flask._helper import FlaskSpanHandler, FlaskResponseSpanHandler
from monocle_apptrace.instrumentation.metamodel.requests.methods import (REQUESTS_METHODS, )
from monocle_apptrace.instrumentation.metamodel.requests._helper import RequestSpanHandler
from monocle_apptrace.instrumentation.metamodel.httpx.methods import (HTTPX_METHODS, )
from monocle_apptrace.instrumentation.metamodel.httpx._helper import HttpxSpanHandler
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
from monocle_apptrace.instrumentation.metamodel.agentcore.agentcore_handler import AgentCoreSpanHandler
from monocle_apptrace.instrumentation.metamodel.lambdafunc._helper import lambdaSpanHandler
from monocle_apptrace.instrumentation.metamodel.lambdafunc.methods import LAMBDA_HTTP_METHODS
from monocle_apptrace.instrumentation.metamodel.mcp.methods import MCP_METHODS
from monocle_apptrace.instrumentation.metamodel.mcp.mcp_processor import MCPAgentHandler
from monocle_apptrace.instrumentation.metamodel.a2a.methods import A2A_CLIENT_METHODS
from monocle_apptrace.instrumentation.metamodel.litellm.methods import LITELLM_METHODS
from monocle_apptrace.instrumentation.metamodel.litellm.litellm_span_handler import LiteLLMSyncSpanHandler
from monocle_apptrace.instrumentation.metamodel.adk.methods import ADK_METHODS
from monocle_apptrace.instrumentation.metamodel.adk.adk_handler import AdkSpanHandler
from monocle_apptrace.instrumentation.metamodel.mistral.methods import MISTRAL_METHODS
from monocle_apptrace.instrumentation.metamodel.strands.methods import STRAND_METHODS
from monocle_apptrace.instrumentation.metamodel.typesafe.methods import TYPESAFE_METHODS
from monocle_apptrace.instrumentation.metamodel.strands.strands_processor import StrandsSpanHandler
from monocle_apptrace.instrumentation.metamodel.agentcore.methods import AGENTCORE_METHODS
from monocle_apptrace.instrumentation.metamodel.claude_cli.methods import CLAUDE_CLI_PROXY_METHODS
from monocle_apptrace.instrumentation.metamodel.codex_cli.codex_span_handler import CodexSpanHandler
from monocle_apptrace.instrumentation.metamodel.codex_cli.methods import CODEX_CLI_PROXY_METHODS
from monocle_apptrace.instrumentation.metamodel.github_copilot.github_copilot_span_handler import GitHubCopilotSpanHandler
from monocle_apptrace.instrumentation.metamodel.github_copilot.methods import GITHUB_COPILOT_PROXY_METHODS

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

# Keyed by metamodel folder name, so users can pick a subset.
METAMODEL_METHODS: Dict[str, list] = {
    "langchain": LANGCHAIN_METHODS,
    "llamaindex": LLAMAINDEX_METHODS,
    "haystack": HAYSTACK_METHODS,
    "botocore": BOTOCORE_METHODS,
    "flask": FLASK_METHODS,
    "requests": REQUESTS_METHODS,
    "httpx": HTTPX_METHODS,
    "langgraph": LANGGRAPH_METHODS,
    "crew_ai": CREW_AI_METHODS,
    "msagent": MSAGENT_METHODS,
    "agents": AGENTS_METHODS,
    "openai": OPENAI_METHODS,
    "teamsai": TEAMAI_METHODS,
    "anthropic": ANTHROPIC_METHODS,
    "aiohttp": AIOHTTP_METHODS,
    "azureaiinference": AZURE_AI_INFERENCE_METHODS,
    "azfunc": AZFUNC_HTTP_METHODS,
    "gemini": GEMINI_METHODS,
    "fastapi": FASTAPI_METHODS,
    "fastmcp": FASTMCP_METHODS,
    "lambdafunc": LAMBDA_HTTP_METHODS,
    "mcp": MCP_METHODS,
    "a2a": A2A_CLIENT_METHODS,
    "litellm": LITELLM_METHODS,
    "adk": ADK_METHODS,
    "mistral": MISTRAL_METHODS,
    "hugging_face": HUGGING_FACE_METHODS,
    "typesafe": TYPESAFE_METHODS,
    "strands": STRAND_METHODS,
    "agentcore": AGENTCORE_METHODS,
    "claude_cli": CLAUDE_CLI_PROXY_METHODS,
    "codex_cli": CODEX_CLI_PROXY_METHODS,
    "github_copilot": GITHUB_COPILOT_PROXY_METHODS,
}

DEFAULT_METHODS_LIST = [m for methods in METAMODEL_METHODS.values() for m in methods]


logger = logging.getLogger(__name__)


def get_supported_metamodels() -> List[str]:
    """Return the names accepted by ``setup_monocle_telemetry(metamodels=...)``."""
    return sorted(METAMODEL_METHODS)


def parse_metamodels(metamodels: Union[str, List[str], None]) -> Optional[List[str]]:
    """Split a comma-separated string into names. A list or None is returned as is."""
    if isinstance(metamodels, str):
        return [name.strip() for name in metamodels.split(",") if name.strip()]
    return metamodels


def validate_metamodels(metamodels: Union[str, List[str], None]) -> Optional[List[str]]:
    """Drop unknown names with a warning. Falls back to all metamodels (None) if no name is valid."""
    metamodels = parse_metamodels(metamodels)
    if metamodels is None:
        return None
    valid = []
    for name in dict.fromkeys(metamodels):
        if name in METAMODEL_METHODS:
            valid.append(name)
            continue
        suggestion = get_close_matches(name, METAMODEL_METHODS, n=1)
        hint = f" Did you mean '{suggestion[0]}'?" if suggestion else ""
        logger.warning(f"Unknown metamodel '{name}' is ignored.{hint} Valid names: {get_supported_metamodels()}")
    if metamodels and not valid:
        logger.warning("None of the given metamodels are valid, so all metamodels are loaded.")
        return None
    return valid


def get_metamodel_methods(metamodels: Union[str, List[str], None] = None) -> list:
    """Return wrapper methods for the given metamodel names, or all of them when None."""
    metamodels = parse_metamodels(metamodels)
    if metamodels is None:
        return DEFAULT_METHODS_LIST
    return [m for name in dict.fromkeys(metamodels) for m in METAMODEL_METHODS.get(name, [])]

MONOCLE_SPAN_HANDLERS: Dict[str, SpanHandler] = {
    "default": SpanHandler(),
    "aiohttp_handler": aiohttpSpanHandler(),
    "botocore_handler": BotoCoreSpanHandler(),
    "flask_handler": FlaskSpanHandler(),
    "flask_response_handler": FlaskResponseSpanHandler(),
    "request_handler": RequestSpanHandler(),
    "httpx_handler": HttpxSpanHandler(),
    "non_framework_handler": NonFrameworkSpanHandler(),
    "openai_handler": OpenAISpanHandler(),
    "openai_agents_handler": OpenAIAgentsSpanHandler(),
    "azure_func_handler": azureSpanHandler(),
    "mcp_agent_handler": MCPAgentHandler(),
    "fastapi_handler": FastAPISpanHandler(),
    "fastapi_response_handler": FastAPIResponseSpanHandler(),
    "langgraph_agent_handler": LanggraphAgentHandler(),
    "langgraph_tool_handler": LanggraphToolHandler(),
    "crew_ai_agent_handler": CrewAIAgentHandler(),
    "crew_ai_task_handler": CrewAITaskHandler(),
    "crew_ai_tool_handler": CrewAIToolHandler(),
    "msagent_request_handler": MSAgentRequestHandler(),
    "msagent_agent_handler": MSAgentAgentHandler(),
    "msagent_inference_handler": MSAgentInferenceHandler(),
    "msagent_inference_stream_handler": MSAgentInferenceStreamHandler(),
    "msagent_tool_handler": MSAgentToolHandler(),
    "msagent_orchestrator_handler": MSAgentOrchestratorHandler(),
    "agents_agent_handler": AgentsSpanHandler(),
    "llamaindex_tool_handler": LlamaIndexToolHandler(),
    "llamaindex_agent_handler": LlamaIndexAgentHandler(),
    "llamaindex_workflow_handler": LlamaIndexWorkflowHandler(),
    "llamaindex_single_agent_tool_handler": LlamaIndexSingleAgenttToolHandlerWrapper(),
    "lambda_func_handler": lambdaSpanHandler(),
    "agentcore_handler": AgentCoreSpanHandler(),
    "adk_handler": AdkSpanHandler(),
    "strands_handler": StrandsSpanHandler(),
    "claude_handler": ClaudeSpanHandler(),
    "codex_handler": CodexSpanHandler(),
    "github_copilot_handler": GitHubCopilotSpanHandler(),
    "litellm_sync_handler": LiteLLMSyncSpanHandler()
}
