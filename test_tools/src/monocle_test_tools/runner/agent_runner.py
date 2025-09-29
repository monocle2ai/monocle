from abc import abstractmethod
from typing import Any
from proto import Enum

class AgentTypes(Enum, str):
    GOOGLE_ADK = "google_adk"
    OPENAI = "openai"
    LANGGRAPH = "langgraph"

class AgentRunner:
    @abstractmethod
    async def run_agent_async(root_agent, test_message: Any):
        raise NotImplementedError("This is a placeholder function. Please implement the function in your test setup.")

    @abstractmethod
    def run_agent(root_agent, test_message: Any):
        raise NotImplementedError("This is a placeholder function. Please implement the function in your test setup.")

def get_agent_runner(runner_type: str) -> AgentRunner:
    if runner_type == AgentTypes.GOOGLE_ADK:
        from .adk_runner import ADKRunner
        return ADKRunner()
    elif runner_type == AgentTypes.OPENAI:
        from .openai_runner import OpenAIRunner
        return OpenAIRunner()
    elif runner_type == AgentTypes.LANGGRAPH:
        from .lg_runner import LGRunner
        return LGRunner()
    else:
        raise ValueError(f"Unknown runner type: {runner_type}")
