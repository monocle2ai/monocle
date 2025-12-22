import logging
from typing import Any, Union
from monocle_test_tools.runner.agent_runner import AgentRunner

logger = logging.getLogger(__name__)

class CrewAIRunner(AgentRunner):
    """Runner for CrewAI crews and agents."""
    
    async def run_agent_async(self, crew, request: Union[str, dict, Any]):
        """
        Execute a CrewAI crew asynchronously.
        
        Args:
            crew: The CrewAI Crew instance to execute
            request: The input to the crew. Can be:
                - str: Simple text request (will be wrapped in inputs dict)
                - dict: Direct inputs dictionary for the crew
                - Any: Other input format accepted by the crew
        
        Returns:
            The result from the crew execution (typically a string)
        """
        # If request is a simple string, wrap it in inputs dict
        if isinstance(request, str):
            inputs = {"request": request}
        elif isinstance(request, dict):
            inputs = request
        else:
            # For other types, try to use it directly
            inputs = request
        
        # Execute the crew asynchronously
        result = await crew.kickoff_async(inputs=inputs)
        
        logger.debug(f"CrewAI result: {result}")
        
        # Return string representation of result
        return str(result) if result is not None else None
    
    def run_agent(self, crew, request: Union[str, dict, Any]):
        """
        Execute a CrewAI crew synchronously.
        
        Args:
            crew: The CrewAI Crew instance to execute
            request: The input to the crew. Can be:
                - str: Simple text request (will be wrapped in inputs dict)
                - dict: Direct inputs dictionary for the crew
                - Any: Other input format accepted by the crew
        
        Returns:
            The result from the crew execution (typically a string)
        """
        # If request is a simple string, wrap it in inputs dict
        if isinstance(request, str):
            inputs = {"request": request}
        elif isinstance(request, dict):
            inputs = request
        else:
            # For other types, try to use it directly
            inputs = request
        
        # Execute the crew synchronously
        result = crew.kickoff(inputs=inputs)
        
        logger.debug(f"CrewAI result: {result}")
        
        # Return string representation of result
        return str(result) if result is not None else None
