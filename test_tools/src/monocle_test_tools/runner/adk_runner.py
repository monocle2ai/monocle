
from typing import Union
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
import logging
logger = logging.getLogger(__name__)
from agent_runner import AgentRunner

APP_NAME = "monocle_test"
USER_ID = "monocle_test_user"
SESSION_ID = "monocle_test_session"

class ADKRunner(AgentRunner):
    async def run_agent(self, root_agent, test_message: Union[types.Content, str]):
        session_service = InMemorySessionService()
        runner = Runner(
            agent=root_agent,
            app_name=APP_NAME,
        session_service=session_service
        )
        await session_service.create_session(
            app_name=APP_NAME, 
            user_id=USER_ID,
            session_id=SESSION_ID
        )
        if isinstance(test_message, str):
            content = types.Content(role='user', parts=[types.Part(text=test_message)])
        else:
            content = test_message
        # Process events as they arrive using async for
        content = None
        async for event in runner.run_async(
            user_id=USER_ID,
            session_id=SESSION_ID,
            new_message=content
        ):
            # For final response
            if event.is_final_response():
                content = event.content
                logger.debug(event.content)  # End line after response
        return content
        