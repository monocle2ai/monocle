from agent_runner import AgentRunner
from typing import Any, Union
import logging
logger = logging.getLogger(__name__)
from agents import Runner

class OpenAIRunner(AgentRunner):
    async def run_agent_async(self, root_agent, request: Any):
        # Test the multi-agent workflow with weather information
        result = await Runner.run(
            root_agent,
            request,
        )