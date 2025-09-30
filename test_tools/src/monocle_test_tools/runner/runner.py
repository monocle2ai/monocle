from enum import Enum
from monocle_test_tools.runner.agent_runner import AgentRunner

class AgentTypes(str, Enum):
    GOOGLE_ADK = "google_adk"
    OPENAI = "openai"
    LANGGRAPH = "langgraph"
    
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