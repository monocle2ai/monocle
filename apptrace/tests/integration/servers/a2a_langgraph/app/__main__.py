import logging
import os
import sys

import click
import httpx
import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)
from dotenv import load_dotenv

from agent import CurrencyAgent
from agent_executor import CurrencyAgentExecutor

from a2a.server.tasks import (
    InMemoryPushNotificationConfigStore,
    BasePushNotificationSender,
)


load_dotenv()

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class MissingAPIKeyError(Exception):
    """Exception for missing API key."""


@click.command()
@click.option("--host", "host", default="localhost")
@click.option("--port", "port", default=10000)
def main(host, port):
    """Starts the Currency Agent server."""
    try:
        if os.getenv("model_source", "google") == "google":
            if not os.getenv("GOOGLE_API_KEY"):
                raise MissingAPIKeyError("GOOGLE_API_KEY environment variable not set.")
        else:
            if not os.getenv("TOOL_LLM_URL"):
                raise MissingAPIKeyError("TOOL_LLM_URL environment variable not set.")
            if not os.getenv("TOOL_LLM_NAME"):
                raise MissingAPIKeyError(
                    "TOOL_LLM_NAME environment not variable not set."
                )

        capabilities = AgentCapabilities(streaming=True, pushNotifications=True)
        skill = AgentSkill(
            id="convert_currency",
            name="Currency Exchange Rates Tool",
            description="Helps with exchange values between various currencies",
            tags=["currency conversion", "currency exchange"],
            examples=["What is exchange rate between USD and GBP?"],
        )
        agent_card = AgentCard(
            name="Currency Agent",
            description="Helps with exchange rates for currencies",
            url=f"http://{host}:{port}/",
            version="1.0.0",
            defaultInputModes=CurrencyAgent.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=CurrencyAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[skill],
        )

        # --8<-- [start:DefaultRequestHandler]
        httpx_client = httpx.AsyncClient()
        push_notification_config_store = InMemoryPushNotificationConfigStore()
        push_notification_sender = BasePushNotificationSender(
            httpx_client, config_store=push_notification_config_store
        )
        request_handler = DefaultRequestHandler(
            agent_executor=CurrencyAgentExecutor(),
            task_store=InMemoryTaskStore(),
            push_config_store=push_notification_config_store,
            push_sender=push_notification_sender,
        )
        server = A2AStarletteApplication(
            agent_card=agent_card, http_handler=request_handler
        )

        uvicorn.run(server.build(), host=host, port=port)
        # --8<-- [end:DefaultRequestHandler]

    except MissingAPIKeyError as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred during server startup: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
