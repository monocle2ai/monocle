from abc import abstractmethod
from typing import Any

class AgentRunner:
    @abstractmethod
    async def run_agent_async(self, root_agent,*args, **kwargs) -> Any:
        raise NotImplementedError("This is a placeholder function. Please implement the function in your test setup.")

    @abstractmethod
    def run_agent(self, root_agent, *args, **kwargs) -> Any:
        raise NotImplementedError("This is a placeholder function. Please implement the function in your test setup.")
    
    def get_remote_traces_source(self) -> str:
        """Check if the runner has remote traces. This can be overridden if the runner needs to fetch traces in a specific way."""
        return None

