import asyncio
from typing import Any, Union
import logging
logger = logging.getLogger(__name__)
from monocle_test_tools.runner.agent_runner import AgentRunner

class LlamaIndexRunner(AgentRunner):
    """Runner for LlamaIndex agents."""
    
    async def run_agent_async(self, root_agent, test_message: Union[str, Any]):
        """
        Run a LlamaIndex agent asynchronously.
        
        Args:
            root_agent: The LlamaIndex agent (e.g., ReActAgent, QueryEngine, AgentWorkflow, etc.)
                        OR a callable function that creates and runs the agent
            test_message: The input message/query for the agent
            
        Returns:
            The agent's response
        """
        try:
            # Check if root_agent is a callable function (wrapper function)
            if callable(root_agent) and not hasattr(root_agent, 'achat') and not hasattr(root_agent, 'run'):
                # It's a wrapper function, call it directly with the message
                result = root_agent(test_message)
                # If it returns a coroutine, await it
                if asyncio.iscoroutine(result):
                    return await result
                return result
            
            # Check for AgentWorkflow which uses user_msg parameter
            agent_type = type(root_agent).__name__
            
            # LlamaIndex agents typically have an async chat or run method
            if agent_type == 'AgentWorkflow' and hasattr(root_agent, 'run'):
                # AgentWorkflow.run() is awaitable and uses user_msg parameter
                response = await root_agent.run(user_msg=test_message)
                
            elif hasattr(root_agent, 'achat'):
                response = await root_agent.achat(test_message)
            elif hasattr(root_agent, 'arun'):
                response = await root_agent.arun(test_message)
            elif hasattr(root_agent, 'aquery'):
                response = await root_agent.aquery(test_message)
            elif hasattr(root_agent, 'run'):
                # If no async method, run sync method in executor
                response = await asyncio.get_event_loop().run_in_executor(
                    None, root_agent.run, test_message
                )
            elif hasattr(root_agent, 'chat'):
                # For chat-based agents
                response = await asyncio.get_event_loop().run_in_executor(
                    None, root_agent.chat, test_message
                )
            elif hasattr(root_agent, 'query'):
                # For query engines
                response = await asyncio.get_event_loop().run_in_executor(
                    None, root_agent.query, test_message
                )
            else:
                raise AttributeError(
                    f"Agent {agent_type} does not have a recognized execution method "
                    "(achat, arun, aquery, run, chat, or query)"
                )
            
            logger.debug(f"LlamaIndex agent response: {response}")
            logger.debug(f"Response type: {type(response)}, Response attributes: {dir(response)}")
            
            # Extract text from response object if it's not already a string
            if isinstance(response, str):
                return response
            elif hasattr(response, 'response'):
                return str(response.response)
            elif hasattr(response, 'message'):
                if hasattr(response.message, 'content'):
                    return str(response.message.content)
                return str(response.message)
            elif hasattr(response, 'text'):
                return str(response.text)
            elif isinstance(response, dict):
                # Try to extract from dict
                if 'response' in response:
                    return str(response['response'])
                elif 'text' in response:
                    return str(response['text'])
                elif 'output' in response:
                    return str(response['output'])
                else:
                    return str(response)
            else:
                return str(response)
                
        except Exception as e:
            logger.error(f"Error running LlamaIndex agent: {e}")
            raise
    
    def run_agent(self, root_agent, test_message: Union[str, Any]):
        """
        Run a LlamaIndex agent synchronously.
        
        Args:
            root_agent: The LlamaIndex agent (e.g., ReActAgent, QueryEngine, etc.)
            test_message: The input message/query for the agent
            
        Returns:
            The agent's response
        """
        try:
            # LlamaIndex agents typically have chat or run methods
            if hasattr(root_agent, 'chat'):
                response = root_agent.chat(test_message)
            elif hasattr(root_agent, 'run'):
                response = root_agent.run(test_message)
            elif hasattr(root_agent, 'query'):
                response = root_agent.query(test_message)
            else:
                raise AttributeError(
                    f"Agent {type(root_agent).__name__} does not have a recognized execution method "
                    "(chat, run, or query)"
                )
            
            logger.debug(f"LlamaIndex agent response: {response}")
            logger.debug(f"Response type: {type(response)}")
            
            # Extract text from response object if it's not already a string
            if isinstance(response, str):
                return response
            elif hasattr(response, 'response'):
                return str(response.response)
            elif hasattr(response, 'message'):
                if hasattr(response.message, 'content'):
                    return str(response.message.content)
                return str(response.message)
            elif hasattr(response, 'text'):
                return str(response.text)
            elif isinstance(response, dict):
                # Try to extract from dict
                if 'response' in response:
                    return str(response['response'])
                elif 'text' in response:
                    return str(response['text'])
                elif 'output' in response:
                    return str(response['output'])
                else:
                    return str(response)
            else:
                return str(response)
                
        except Exception as e:
            logger.error(f"Error running LlamaIndex agent: {e}")
            raise
