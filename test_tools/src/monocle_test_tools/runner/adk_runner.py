from typing import Union
import uuid
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
import logging
logger = logging.getLogger(__name__)
from monocle_test_tools.runner.agent_runner import AgentRunner

APP_NAME = "monocle_test"
USER_ID = "monocle_test_user"
SESSION_ID = f"monocle_test_session_{uuid.uuid4().hex}"

class ADKRunner(AgentRunner):
    async def run_agent_async(self, root_agent, *args, session_id: str = None):
        # Extract request from args - it should be the first positional argument
        test_message = args[0] if args else None

        if session_id is None:
            session_id = SESSION_ID
            
        session_service = InMemorySessionService()
        runner = Runner(
            agent=root_agent,
            app_name=APP_NAME,
        session_service=session_service
        )
        await session_service.create_session(
            app_name=APP_NAME, 
            user_id=USER_ID,
            session_id=session_id
        )
        if isinstance(test_message, str):
            content = types.Content(role='user', parts=[types.Part(text=test_message)])
        else:
            content = test_message
        # Process events as they arrive using async for
        async for event in runner.run_async(
            user_id=USER_ID,
            session_id=session_id,
            new_message=content
        ):
            # For final response
            if event.is_final_response():
                content = event.content
                logger.debug(event.content)  # End line after response
        if content is not None:
            return content.parts[0].text
        else:
            return None