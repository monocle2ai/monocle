
import asyncio
import time
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent
from llama_index.llms.openai import OpenAI
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from llama_index.core.agent import ReActAgent
from llama_index.tools.mcp import aget_tools_from_mcp_url
import logging

logger = logging.getLogger(__name__)

# setup_monocle_telemetry(workflow_name="travel-agent-lmx-wf-05", monocle_exporters_list='file')

def book_hotel(hotel_name: str):
    """Book a hotel"""
    return f"Successfully booked a stay at {hotel_name}."

def book_flight(from_airport: str, to_airport: str):
    """Book a flight"""
    return f"Successfully booked a flight from {from_airport} to {to_airport}."

async def setup_agents():
    
    async def get_mcp_tools():
        """Get tools from the MCP weather server."""
        try:
            weather_tools = await aget_tools_from_mcp_url(
                "http://localhost:8001/weather/mcp/"
            )
            return weather_tools
        except Exception as e:
            logger.warning(f"Failed to get MCP tools: {e}")
            return []

    # Get MCP weather tools
    weather_tools = await get_mcp_tools()
   
    llm = OpenAI(model="gpt-4o")

    flight_tool = FunctionTool.from_defaults(
        fn=book_flight,
        name="lmx_book_flight_tool_05",
        description="Books a flight from one airport to another."
    )
    flight_agent = FunctionAgent(
                                name="lmx_flight_booking_agent_05",
                                tools=[flight_tool],
                                llm=llm,
                                system_prompt="""You are a flight booking agent who books flights as per the request. 
                                        When you receive a flight booking request, immediately use the book_flight tool to complete the booking.
                                        After successfully booking the flight, you MUST handoff back to lmx_coordinator_05 with the booking confirmation message.""",
                                description="Flight booking agent",
                                can_handoff_to=["lmx_coordinator_05"]
                            )

    hotel_tool = FunctionTool.from_defaults(
        fn=book_hotel,
        name="lmx_book_hotel_tool_05",
        description="Books a hotel stay."
    )
    hotel_agent = FunctionAgent(
                                name="lmx_hotel_booking_agent_05",
                                tools=[hotel_tool],
                                llm=llm,
                                system_prompt="""You are a hotel booking agent who books hotels as per the request.
                                        When you receive a hotel booking request, immediately use the book_hotel tool to complete the booking.
                                        After successfully booking the hotel, you MUST handoff back to lmx_coordinator_05 with the booking confirmation message.""",
                                description="Hotel booking agent",
                                can_handoff_to=["lmx_coordinator_05"]
                            )

    coordinator = FunctionAgent(
                                name="lmx_coordinator_05",
                                tools = weather_tools,
                                llm=llm,
                                system_prompt=
                                """You are a coordinator agent who manages flight and hotel booking agents. 
                         
                                    For each user request:
                                    1. First delegate flight booking to the lmx_flight_booking_agent_05 agent
                                    2. After flight booking is complete, delegate hotel booking to the lmx_hotel_booking_agent_05 agent  
                                    3. Once both bookings are complete, use weather tools if weather information is requested
                                    4. Provide a consolidated response with all booking details and weather information
                                    
                                    Always ensure both agents complete their tasks and gather all information before providing the final response.
                                    Continue delegating until all tasks are done.""",
                                description="Travel booking coordinator agent",
                                can_handoff_to=["lmx_flight_booking_agent_05", "lmx_hotel_booking_agent_05"])

    agent_workflow = AgentWorkflow(
        handoff_prompt="""As soon as you have figured out the requirements, 
        If you need to delegate the task to another agent, then delegate the task to that agent.
        For eg if you need to book a flight, then delegate the task to flight agent.
        If you need to book a hotel, then delegate the task to hotel agent.
        If you can book hotel or flight direclty, then do that and collect the response and then handoff to the supervisor agent.
        {agent_info}
        """,
        agents=[coordinator, flight_agent, hotel_agent],
        root_agent=coordinator.name
    )
    return agent_workflow

async def run_agent(user_msg: str = None):
    """Test multi-agent interaction with flight and hotel booking."""

    agent_workflow = await setup_agents()
    
    # If no user_msg provided, use default requests (for manual testing)
    if user_msg is None:
        requests = [
            "Book a flight from San Jose to Boston and a book hotel stay at Hyatt Hotel, and tell the weather in Boston.",
    #        "book a flight from San Francisco to New York and a book hotel stay at Hilton Hotel",
    #        "book a flight from Los Angeles to Miami and a book hotel stay at Marriott Hotel",
        ]
        for req in requests:
            resp = await agent_workflow.run(user_msg=req)
            print(resp)
        return None
    else:
        # For test framework: process single message and return response as string
        resp = await agent_workflow.run(user_msg=user_msg)
        # Extract the string response from the workflow result
        if hasattr(resp, 'response'):
            return str(resp.response)
        elif isinstance(resp, dict) and 'response' in resp:
            return str(resp['response'])
        else:
            return str(resp)

async def setup_react_agent():
    """Create a simple ReActAgent for testing achat/arun methods."""
    llm = OpenAI(model="gpt-4o")
    
    flight_tool = FunctionTool.from_defaults(
        fn=book_flight,
        name="lmx_book_flight_tool_react",
        description="Books a flight from one airport to another."
    )
    
    hotel_tool = FunctionTool.from_defaults(
        fn=book_hotel,
        name="lmx_book_hotel_tool_react",
        description="Books a hotel stay."
    )
    
    agent = ReActAgent.from_tools(
        [flight_tool, hotel_tool],
        llm=llm,
        verbose=True
    )
    return agent

async def run_react_agent_achat(user_msg: str):
    """Test ReActAgent with achat method (async chat)."""
    agent = await setup_react_agent()
    response = await agent.achat(user_msg)
    
    if hasattr(response, 'response'):
        return str(response.response)
    else:
        return str(response)

async def run_react_agent_aquery(user_msg: str):
    """Test ReActAgent with aquery method if available."""
    agent = await setup_react_agent()
    
    # Some agents have aquery method
    if hasattr(agent, 'aquery'):
        response = await agent.aquery(user_msg)
    else:
        # Fallback to achat
        response = await agent.achat(user_msg)
    
    if hasattr(response, 'response'):
        return str(response.response)
    else:
        return str(response)

def run_react_agent_chat(user_msg: str):
    """Test ReActAgent with synchronous chat method."""
    llm = OpenAI(model="gpt-4o")
    
    flight_tool = FunctionTool.from_defaults(
        fn=book_flight,
        name="lmx_book_flight_tool_sync",
        description="Books a flight from one airport to another."
    )
    
    agent = ReActAgent.from_tools([flight_tool], llm=llm, verbose=True)
    response = agent.chat(user_msg)
    
    if hasattr(response, 'response'):
        return str(response.response)
    else:
        return str(response)

def run_query_engine(user_msg: str):
    """Test QueryEngine with synchronous query method."""
    from llama_index.core import VectorStoreIndex, Document
    
    # Create simple documents
    documents = [
        Document(text="Flight booking service is available from 9 AM to 5 PM."),
        Document(text="Hotel reservations can be made online or by phone."),
    ]
    
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    
    response = query_engine.query(user_msg)
    
    if hasattr(response, 'response'):
        return str(response.response)
    else:
        return str(response)

async def run_query_engine_async(user_msg: str):
    """Test QueryEngine with asynchronous aquery method."""
    from llama_index.core import VectorStoreIndex, Document
    
    # Create simple documents
    documents = [
        Document(text="Flight booking service is available from 9 AM to 5 PM."),
        Document(text="Hotel reservations can be made online or by phone."),
    ]
    
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    
    response = await query_engine.aquery(user_msg)
    
    if hasattr(response, 'response'):
        return str(response.response)
    else:
        return str(response)

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARN)
    setup_monocle_telemetry(workflow_name="travel-agent-lmx-wf-05", monocle_exporters_list='file')
    asyncio.run(run_agent())
