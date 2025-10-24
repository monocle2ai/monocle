import logging
from typing import Union
from monocle_test_tools.runner.agent_runner import AgentRunner
logger = logging.getLogger(__name__)

class LGRunner(AgentRunner):
    async def run_agent_async(self, root_agent, request: Union[str, dict]):
        if isinstance(request, str):
            input = {
                "messages": [
                {
                    "role": "user",
                    "content": request
                }
            ]
        }
        else:
            input = request
        chunk = await root_agent.ainvoke(input=input)
        logger.debug(chunk["messages"][-1].content)
        return chunk["messages"][-1].content

    def run_agent(self, lg_agent, request: Union[str, dict]):
        if isinstance(request, str):
            request = {
                "messages": [
                    {
                        "role": "user",
                        "content": request
                    }
                ]
            }
        else:
            input = request
        chunk = lg_agent.invoke(input=input)
        logger.debug(chunk["messages"][-1].content)
        return chunk["messages"][-1].content