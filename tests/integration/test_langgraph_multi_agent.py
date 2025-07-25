import asyncio
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, BatchSpanProcessor
import uvicorn
from monocle_apptrace.exporters.file_exporter import FileSpanExporter
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langchain_mcp_adapters.client import MultiServerMCPClient
import pytest
import logging
from servers.mcp.weather_server import app
from typing import Optional
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from a2a.client import A2ACardResolver, A2AClient
from langchain_core.tools import BaseTool, tool
from langchain_core.tools.base import ArgsSchema
from pydantic import BaseModel, Field
import httpx
from uuid import uuid4

logger = logging.getLogger(__name__)

import time
import threading

memory_exporter = InMemorySpanExporter()
span_processors = [
    SimpleSpanProcessor(memory_exporter),
    BatchSpanProcessor(FileSpanExporter()),
]
server_thread = None


# pytest start a fast api server once
@pytest.fixture(scope="module")
@pytest.mark.asyncio
async def start_fastapi_server():
    """Start the FastAPI server."""


@pytest.fixture(scope="module")
def setup():
    memory_exporter.clear()
    setup_monocle_telemetry(
        workflow_name="langchain_agent_1",
        span_processors=span_processors,
    )

    # def run_server():
    #     uvicorn.run(app, host="127.0.0.1", port=8001, log_level="info")

    # server_thread = threading.Thread(target=run_server, daemon=True)
    # server_thread.start()
    # time.sleep(0.1)  # time for the server to start

    # yield

    # Note: daemon threads will be terminated when the main process exits


def book_hotel(hotel_name: str):
    """Book a hotel"""
    return f"Successfully booked a stay at {hotel_name} for 50 USD."


def book_flight(from_airport: str, to_airport: str):
    """Book a flight"""
    return (
        f"Successfully booked a flight from {from_airport} to {to_airport} for 100 USD."
    )


class CurrencyConversionInput(BaseModel):
    message: str = Field(description="The currencies in prompt like : 'Give exchange rate for USD to EUR'")


class CurrencyConversionTool(BaseTool):
    name: str = "CurrencyConversion"
    description: str = (
        "Gives currency conversion rate"
    )
    args_schema: Optional[ArgsSchema] = CurrencyConversionInput
    return_direct: bool = False
    a2a_client: Optional[A2AClient] = None

    def __init__(self, a2a_client, **kwargs):
        super().__init__(**kwargs)
        self.a2a_client = a2a_client

    def _run(
        self, message: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool synchronously - delegates to async implementation."""
        import asyncio

        try:
            # Run the async version in a new event loop if none exists
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, we can't use asyncio.run
                # This is a simplified sync version for demonstration
                return f"Currency A2A Client would send message: '{message}'"
            else:
                return asyncio.run(self._arun(message, run_manager))
        except RuntimeError:
            return asyncio.run(self._arun(message, run_manager))

    async def _arun(
        self,
        message: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously to communicate with currency A2A agent."""
        try:
            if self.a2a_client is None:
                return "Currency A2A Client not initialized"
            
            # Import A2A classes
            from a2a.types import MessageSendParams, SendMessageRequest
            
            # Prepare message payload
            send_message_payload = {
                'message': {
                    'role': 'user',
                    'parts': [
                        {'kind': 'text', 'text': message}
                    ],
                    'messageId': uuid4().hex,
                },
            }
            
            request = SendMessageRequest(
                id=str(uuid4()), 
                params=MessageSendParams(**send_message_payload)
            )
            
            # Send message using the pre-created client
            response = await self.a2a_client.send_message(request)
            # parts = response.root.result.parts[0].root
            return f"Currency A2A Response: {response.model_dump(mode='json', exclude_none=True)}"
                
        except ImportError as e:
            return f"A2A Client libraries not available: {str(e)}"
        except Exception as e:
            return f"Currency A2A Client error: {str(e)}"


async def createCurrencyA2ATool(base_url: str = "http://localhost:10000"):
    """Factory method to create a CurrencyA2ATool with pre-initialized A2A client."""
    try:
        # Import A2A classes
        from a2a.client import A2ACardResolver, A2AClient
        
        # Create a persistent httpx client
        httpx_client = httpx.AsyncClient()
        
        try:
            # Initialize A2ACardResolver
            resolver = A2ACardResolver(
                httpx_client=httpx_client,
                base_url=base_url,
            )

            # Fetch agent card
            agent_card = await resolver.get_agent_card()

            # Create A2A client with the httpx client and agent card
            a2a_client = A2AClient(
                httpx_client=httpx_client, 
                agent_card=agent_card
            )

            # Create and return the tool with the pre-initialized A2A client
            return CurrencyConversionTool(a2a_client=a2a_client)
            
        except Exception as e:
            # If initialization fails, clean up the httpx client
            await httpx_client.aclose()
            raise e

    except ImportError as e:
        # Return a tool with no client if A2A libraries aren't available
        return CurrencyConversionTool(a2a_client=None)
    except Exception as e:
        # Return a tool with no client if initialization fails
        return CurrencyConversionTool(a2a_client=None)

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

async def setup_agents():

    # Set up MCP client for monocle repo
    async def get_mcp_tools():
        """Get tools from the monocle MCP server."""
        client = MultiServerMCPClient(
            {
                "weather": {
                    "url": "http://localhost:8001/weather/mcp/",
                    "transport": "streamable_http",
                }
            }
        )
        # get list of all tools from the MCP server and their descriptions
        tools = await client.get_tools()
        return tools

    weather_tools = await get_mcp_tools()
    currency_a2a_tool = await createCurrencyA2ATool()
    final_tools = weather_tools + [currency_a2a_tool] + [multiply]

    flight_assistant = create_react_agent(
        model="openai:gpt-4o",
        tools=[book_flight],
        prompt="You are a flight booking assistant",
        name="flight_assistant",
    )

    hotel_assistant = create_react_agent(
        model="openai:gpt-4o",
        tools=[book_hotel],
        prompt="You are a hotel booking assistant",
        name="hotel_assistant",
    )

    supervisor = create_supervisor(
        agents=[flight_assistant, hotel_assistant],
        tools=final_tools,
        model=ChatOpenAI(model="gpt-4o"),
        prompt=(
            """You manage a hotel booking assistant and a flight booking assistant. 
            Assign work to them.
            Use the currency conversion and multiply tool for converting currencies. Don't use approximate conversions."""
        ),
    ).compile()

    return supervisor


@pytest.mark.integration()
@pytest.mark.asyncio
async def test_multi_agent(setup):
    """Test multi-agent interaction with flight and hotel booking."""

    supervisor = await setup_agents()
    chunk = supervisor.invoke(
        input={
            "messages": [
                {
                    "role": "user",
                    "content": """book a flight from BOS to JFK or LAX. And also book me Hyatt hotel at LAX.""",
                }
            ]
        }
    )
    print(chunk)
    print("\n")
    verify_spans()


@pytest.mark.integration()
@pytest.mark.asyncio
async def test_async_multi_agent(setup):
    """Test multi-agent interaction with flight and hotel booking."""

    supervisor = await setup_agents()
    chunk = await supervisor.ainvoke(
        input={
            "messages": [
                {
                    "role": "user",
                    "content": """book a flight from BOS to JFK or LAX, which ever has lower temperature and a hotel stay at McKittrick Hotel at JFK or Sheraton Gateway Hotel at LAX. Give me the cost in INR.""",
                }
            ]
        }
    )
    print(chunk)
    print("\n")
    verify_spans()


def verify_spans():
    time.sleep(2)
    found_inference = found_agent = found_tool = False
    found_flight_agent = found_hotel_agent = found_supervisor_agent = False
    found_book_hotel_tool = found_book_flight_tool = False
    found_book_flight_delegation = found_book_hotel_delegation = False
    spans = memory_exporter.get_finished_spans()
    for span in spans:
        span_attributes = span.attributes

        if "span.type" in span_attributes and (
            span_attributes["span.type"] == "inference"
            or span_attributes["span.type"] == "inference.framework"
        ):
            # Assertions for all inference attributes
            assert span_attributes["entity.1.type"] == "inference.openai"
            assert "entity.1.provider_name" in span_attributes
            assert "entity.1.inference_endpoint" in span_attributes
            assert span_attributes["entity.2.name"] == "gpt-4o"
            assert span_attributes["entity.2.type"] == "model.llm.gpt-4o"

            # Assertions for metadata
            span_input, span_output, span_metadata = span.events
            assert "completion_tokens" in span_metadata.attributes
            assert "prompt_tokens" in span_metadata.attributes
            assert "total_tokens" in span_metadata.attributes
            found_inference = True

        if (
            "span.type" in span_attributes
            and span_attributes["span.type"] == "agentic.invocation"
            and "entity.1.name" in span_attributes
        ):
            assert "entity.1.type" in span_attributes
            assert "entity.1.name" in span_attributes
            assert span_attributes["entity.1.type"] == "agent.langgraph"
            if span_attributes["entity.1.name"] == "flight_assistant":
                found_flight_agent = True
            elif span_attributes["entity.1.name"] == "hotel_assistant":
                found_hotel_agent = True
            elif span_attributes["entity.1.name"] == "supervisor":
                found_supervisor_agent = True
            found_agent = True

        if (
            "span.type" in span_attributes
            and span_attributes["span.type"] == "agentic.tool.invocation"
        ):
            assert "entity.1.type" in span_attributes
            assert "entity.1.name" in span_attributes
            assert span_attributes["entity.1.type"] == "tool.langgraph"
            if span_attributes["entity.1.name"] == "book_flight":
                found_book_flight_tool = True
            elif span_attributes["entity.1.name"] == "book_hotel":
                found_book_hotel_tool = True
            found_tool = True

        if (
            "span.type" in span_attributes
            and span_attributes["span.type"] == "agentic.delegation"
        ):
            assert "entity.1.type" in span_attributes
            assert "entity.1.from_agent" in span_attributes
            assert "entity.1.to_agent" in span_attributes
            assert span_attributes["entity.1.type"] == "agent.langgraph"
            if span_attributes["entity.1.to_agent"] == "flight_assistant":
                found_book_flight_delegation = True
            elif span_attributes["entity.1.to_agent"] == "hotel_assistant":
                found_book_hotel_delegation = True

    assert found_inference, "Inference span not found"
    assert found_agent, "Agent span not found"
    assert found_tool, "Tool span not found"
    assert found_book_flight_delegation, "Book flight delegation span not found"
    assert found_book_hotel_delegation, "Book hotel delegation span not found"
    assert found_flight_agent, "Flight assistant agent span not found"
    assert found_hotel_agent, "Hotel assistant agent span not found"
    assert found_supervisor_agent, "Supervisor agent span not found"
    assert found_book_flight_tool, "Book flight tool span not found"
    assert found_book_hotel_tool, "Book hotel tool span not found"


if __name__ == "__main__":
    pytest.main([__file__, "-s", "--tb=short"])
