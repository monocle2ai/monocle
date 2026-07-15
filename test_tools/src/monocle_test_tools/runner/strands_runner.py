from typing import Union, Any
import logging
logger = logging.getLogger(__name__)
from monocle_test_tools.runner.agent_runner import AgentRunner
from strands.session.file_session_manager import FileSessionManager

class StrandsRunner(AgentRunner):
    """Runner for AWS Strands Agent Core."""
    
    async def run_agent_async(self, root_agent, test_message: Union[str, Any], session_id: str = None):
        """
        Execute an AWS Strands agent asynchronously.
        
        Args:
            root_agent: The AWS Strands Agent instance to execute
            test_message: The input message (typically a string)
            session_id: Optional session ID for conversation continuity
        
        Returns:
            The response text from the agent
        """

        try:
            # AWS Strands agents are called directly with the prompt
            # Include session_id if provided
            if session_id:
                session_manager = FileSessionManager(session_id=session_id)
                result = root_agent(test_message, session_manager=session_manager)
            else:
                result = root_agent(test_message)
            
            # Extract the text content from the response
            if hasattr(result, 'message') and 'content' in result.message:
                response_text = result.message['content'][0]['text']
            else:
                response_text = str(result)
                
            logger.debug(f"AWS Strands agent response: {response_text}")
            return response_text
            
        except Exception as e:
            logger.error(f"Error running AWS Strands agent: {str(e)}")
            raise
    
    def run_agent(self, root_agent, test_message: Union[str, Any], session_id: str = None):
        """
        Execute an AWS Strands agent synchronously.
        
        Args:
            root_agent: The AWS Strands Agent instance to execute
            test_message: The input message (typically a string)
            session_id: Optional session ID for conversation continuity
        
        Returns:
            The response text from the agent
        """
        try:
            # AWS Strands agents are called directly with the prompt
            # Include session_id if provided
            if session_id:
                session_manager = FileSessionManager(session_id=session_id)
                result = root_agent(test_message ,session_manager=session_manager)
            else:
                result = root_agent(test_message)
            
            # Extract the text content from the response
            if hasattr(result, 'message') and 'content' in result.message:
                response_text = result.message['content'][0]['text']
            else:
                response_text = str(result)
                
            logger.debug(f"AWS Strands agent response: {response_text}")
            return response_text
            
        except Exception as e:
            logger.error(f"Error running AWS Strands agent: {str(e)}")
            raise
