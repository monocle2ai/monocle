from monocle_apptrace.instrumentation.common.wrapper import atask_wrapper, task_wrapper
from monocle_apptrace.instrumentation.metamodel.llamaindex.entities.inference import (
    INFERENCE,
)
from monocle_apptrace.instrumentation.metamodel.llamaindex.entities.agent import AGENT, TOOLS, AGENT_REQUEST
from monocle_apptrace.instrumentation.metamodel.llamaindex.entities.retrieval import (
    RETRIEVAL,
)


LLAMAINDEX_METHODS = [
    {
        "package": "llama_index.core.indices.base_retriever",
        "object": "BaseRetriever",
        "method": "retrieve",
        "wrapper_method": task_wrapper,
        "output_processor": RETRIEVAL
    },
    {
        "package": "llama_index.core.indices.base_retriever",
        "object": "BaseRetriever",
        "method": "aretrieve",
        "wrapper_method": atask_wrapper,
        "output_processor": RETRIEVAL
    },
    {
        "package": "llama_index.core.base.base_query_engine",
        "object": "BaseQueryEngine",
        "method": "query",
        "wrapper_method": task_wrapper
    },
    {
        "package": "llama_index.core.base.base_query_engine",
        "object": "BaseQueryEngine",
        "method": "aquery",
        "wrapper_method": atask_wrapper
    },
    {
        "package": "llama_index.core.llms.custom",
        "object": "CustomLLM",
        "method": "chat",
        "wrapper_method": task_wrapper,
        "output_processor": INFERENCE
    },
    {
        "package": "llama_index.core.llms.custom",
        "object": "CustomLLM",
        "method": "achat",
        "wrapper_method": atask_wrapper,
        "output_processor": INFERENCE,
        
    },
    {
        "package": "llama_index.llms.openai.base",
        "object": "OpenAI",
        "method": "chat",
        "wrapper_method": task_wrapper,
        "output_processor": INFERENCE
    },
    {
        "package": "llama_index.llms.openai.base",
        "object": "OpenAI",
        "method": "achat",
        "wrapper_method": atask_wrapper,
        "output_processor": INFERENCE
    },
    {
        "package": "llama_index.llms.mistralai.base",
        "object": "MistralAI",
        "method": "chat",
        "wrapper_method": task_wrapper,
        "output_processor": INFERENCE
    },
    {
        "package": "llama_index.llms.mistralai.base",
        "object": "MistralAI",
        "method": "achat",
        "wrapper_method": atask_wrapper,
        "output_processor": INFERENCE
    },
    {
        "package": "llama_index.core.agent.workflow.multi_agent_workflow",
        "object": "AgentWorkflow",
        "method": "run",
        "span_handler": "llamaindex_agent_handler",
        "wrapper_method": atask_wrapper,
        "output_processor": AGENT_REQUEST
    },
    {
        "package": "llama_index.core.agent",
        "object": "ReActAgent",
        "method": "run",
        "wrapper_method": task_wrapper,
        "span_handler": "llamaindex_agent_handler",
        "output_processor": AGENT
    },
    {
        "package": "llama_index.core.agent.workflow.function_agent",
        "object": "FunctionAgent",
        "method": "finalize",
        "wrapper_method": atask_wrapper,
        "span_handler": "llamaindex_agent_handler",
        "output_processor": AGENT
    },
    {
        "package": "llama_index.core.agent.workflow.function_agent",
        "object": "FunctionAgent",
        "method": "take_step",
        "span_handler": "llamaindex_agent_handler",
        "wrapper_method": atask_wrapper
    },
    {
        "package": "llama_index.core.tools.function_tool",
        "object": "FunctionTool",
        "method": "call",
        "span_handler": "llamaindex_single_agent_tool_handler",
        "wrapper_method": task_wrapper,
        "output_processor": TOOLS
    },
    {
        "package": "llama_index.core.tools.function_tool",
        "object": "FunctionTool",
        "method": "acall",
        "span_handler": "llamaindex_single_agent_tool_handler",
        "wrapper_method": atask_wrapper,
        "output_processor": TOOLS
    },
    {
        "package": "llama_index.core.agent.workflow.multi_agent_workflow",
        "object": "AgentWorkflow",
        "method": "_call_tool",
        "span_handler": "llamaindex_tool_handler",
        "wrapper_method": atask_wrapper,
        "output_processor": TOOLS
    },
    {
        "package": "llama_index.llms.anthropic",
        "object": "Anthropic",
        "method": "chat",
        "wrapper_method": task_wrapper,
        "output_processor": INFERENCE
    },
    {
        "package": "llama_index.llms.anthropic",
        "object": "Anthropic",
        "method": "achat",
        "wrapper_method": atask_wrapper,
        "output_processor": INFERENCE
    },
    {
        "package": "llama_index.llms.gemini",
        "object": "Gemini",
        "method": "chat",
        "wrapper_method": task_wrapper,
        "output_processor": INFERENCE
    },
    {
        "package": "llama_index.llms.gemini",
        "object": "Gemini",
        "method": "achat",
        "wrapper_method": atask_wrapper,
        "output_processor": INFERENCE
    }
]
