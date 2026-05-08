from abc import abstractmethod
from typing import Any

class AgentRunner:
    @abstractmethod
    async def run_agent_async(self, root_agent,*args, **kwargs) -> Any:
        raise NotImplementedError("This is a placeholder function. Please implement the function in your test setup.")

    @abstractmethod
    def run_agent(self, root_agent, *args, **kwargs) -> Any:
        raise NotImplementedError("This is a placeholder function. Please implement the function in your test setup.")

