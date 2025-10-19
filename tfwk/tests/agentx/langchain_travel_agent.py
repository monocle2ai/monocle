import asyncio
import logging
import os
from typing import Annotated, Any, Dict, List, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver  # 👈 UPDATED IMPORT
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import (
    create_react_agent as langgraph_create_react_agent,
)
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up Monocle telemetry for file logging
setup_monocle_telemetry(
    workflow_name="langchain_travel_agent_demo",
    monocle_exporters_list="file"
)


# ==================================================================
# ## 1. Mock Tool Functions
# (Tools remain unchanged)
# ==================================================================

def book_flight_internal(from_city: str, to_city: str, travel_date: str = "2025-12-01", is_business: bool = False) -> Dict[str, Any]:
    """Internal function for flight booking logic."""
    flight_class = "business" if is_business else "economy"
    base_price = 800 if is_business else 400
    popular_destinations = ["Mumbai", "Delhi", "Goa", "Bangalore"]
    if to_city in popular_destinations:
        base_price += 200
    
    booking = {
        "status": "success",
        "booking_id": f"LC-FL-{hash(f'{from_city}-{to_city}') % 10000:04d}",
        "type": "flight",
        "message": f"Successfully booked {flight_class} flight from {from_city} to {to_city} on {travel_date}"
    }
    logger.info(f"Flight booked: {booking['booking_id']}")
    return booking

def book_hotel_internal(hotel_name: str, city: str, check_in_date: str = "2025-12-01", nights: int = 1, is_business: bool = False) -> Dict[str, Any]:
    """Internal function for hotel booking logic."""
    hotel_tier = "luxury" if is_business else "standard"
    
    booking = {
        "status": "success",
        "booking_id": f"LC-HT-{hash(f'{hotel_name}-{city}') % 10000:04d}",
        "type": "hotel",
        "message": f"Successfully booked {nights} nights at {hotel_name} in {city} starting {check_in_date}"
    }
    logger.info(f"Hotel booked: {booking['booking_id']}")
    return booking

def get_travel_recommendations_internal(destination: str) -> Dict[str, Any]:
    """Internal function for recommendations logic."""
    recommendations = {
        "Mumbai": {"attractions": ["Gateway of India", "Marine Drive"], "best_time": "October to March", "tips": "Try local street food"},
        "Delhi": {"attractions": ["Red Fort", "India Gate"], "best_time": "October to April", "tips": "Use metro for transportation"},
        "Goa": {"attractions": ["Beaches", "Old Goa Churches"], "best_time": "November to March", "tips": "Rent a scooter"},
    }
    
    result = {
        "destination": destination,
        "recommendations": recommendations.get(destination, {
            "attractions": ["Local markets"], "best_time": "Check local weather", "tips": "Research local customs"
        }),
        "status": "success"
    }
    logger.info(f"Travel recommendations provided for: {destination}")
    return result

@tool
def book_flight_tool(from_city: str, to_city: str, travel_date: str = "2025-12-01", is_business: bool = False) -> str:
    """Book a flight between cities. Use this for actual flight booking."""
    result = book_flight_internal(from_city, to_city, travel_date, is_business)
    return result["message"]

@tool
def book_hotel_tool(hotel_name: str, city: str, check_in_date: str = "2025-12-01", nights: int = 1, is_business: bool = False) -> str:
    """Book a hotel reservation. Use this for actual hotel booking."""
    result = book_hotel_internal(hotel_name, city, check_in_date, nights, is_business)
    return result["message"]

@tool
def get_recommendations_tool(destination: str) -> str:
    """Get travel recommendations for a destination. Use this for travel advice."""
    result = get_travel_recommendations_internal(destination)
    recommendations = result["recommendations"]
    return f"For {destination}: Attractions - {', '.join(recommendations['attractions'])}. Best time: {recommendations['best_time']}. Tips: {recommendations['tips']}"

ALL_TOOLS = [book_flight_tool, book_hotel_tool, get_recommendations_tool]

# ==================================================================
# ## 2. LangGraph State Definition
# ==================================================================

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    next: str

# ==================================================================
# ## 3. Agent and Router Definitions
# ==================================================================

LLM = ChatOpenAI(model="gpt-4o", temperature=0)

# (Agent definitions using langgraph_create_react_agent remain the same as the user's last provided code)
# NOTE: If you are still encountering the 'system_prompt' or 'prompt' error, you must use the
# manual graph construction from the previous answer. We assume the current structure works for now.

flight_agent = langgraph_create_react_agent(
            model=LLM,
            tools=[book_flight_tool],
            prompt=(
                "You are a helpful flight booking assistant. You can book flights between cities. "
                "Always confirm the travel details and provide flight booking information. "
                "Ask for clarification if travel dates or destinations are unclear."
            ),
            name="Flight_Assistant",
        )

hotel_agent = langgraph_create_react_agent(
            model=LLM,
            tools=[book_hotel_tool],
            prompt=(
                "You are a helpful hotel booking assistant and can book hotels in various cities. "
                "Always confirm accommodation preferences and provide booking details. "
                "Ask for clarification if check-in dates or hotel preferences are unclear, but don't go in circles"
            ),
            name="Hotel_Assistant",
        )

recommendations_agent = langgraph_create_react_agent(
            model=LLM,
            tools=[get_recommendations_tool],
            prompt=(
                "You are a travel recommendations expert. Provide helpful travel advice, "
                "local attractions, best times to visit, and practical tips for destinations. "
                "Be informative and helpful in your recommendations."
            ),
            name="Recommendations_Assistant",
        )


def route_agent(state: AgentState):
    logger.info("Router: Determining next step...")
    
    system_prompt = (
        "You are the Travel Coordinator. Your job is to route the user's request "
        "to the correct specialized agent: 'flight_agent', 'hotel_agent', or 'recommendations_agent'. "
        "If the request is a general greeting, an apology, a summary, or not clearly related to one of the three areas, "
        "route to 'END' to respond directly. "
        "If a specific booking or recommendation is requested, route to the relevant agent."
        "The available agents are: flight_agent, hotel_agent, recommendations_agent."
        "You MUST respond with ONLY the name of the next node (e.g., 'flight_agent', 'hotel_agent', 'END')."
    )
    
    user_message = state["messages"][-1]
    
    # We pass the full history to the router LLM to maintain context for routing
    routing_messages = state["messages"] + [HumanMessage(content=system_prompt)]
    
    response = LLM.invoke(routing_messages)
    route = response.content.strip().lower().replace(' ', '_')
    
    if 'flight' in route:
        next_node = 'flight_agent'
    elif 'hotel' in route:
        next_node = 'hotel_agent'
    elif 'recommendations' in route or 'expert' in route:
        next_node = 'recommendations_agent'
    else:
        next_node = END 
        
    logger.info(f"Router selected: {next_node}")
    return {"next": next_node}



# ==================================================================
# ## 5. Execution Class (Fixed for context)
# ==================================================================

class LangChainTravelAgentDemo:

    def __init__(self):
        self.travel_coordinator = self.__initialize_graph__()
        # Use a fixed thread ID for the entire session
        self.thread_id = "demo-travel-session-1" 

    def __initialize_graph__(self):
                # ==================================================================
        # ## 4. Build the Graph
        # ==================================================================

        graph_builder = StateGraph(AgentState)

        graph_builder.add_node("router", route_agent)
        graph_builder.add_node("flight_agent", flight_agent)
        graph_builder.add_node("hotel_agent", hotel_agent)
        graph_builder.add_node("recommendations_agent", recommendations_agent)

        graph_builder.set_entry_point("router")

        graph_builder.add_conditional_edges(
            "router", 
            lambda x: x["next"], 
            {
                "flight_agent": "flight_agent",
                "hotel_agent": "hotel_agent",
                "recommendations_agent": "recommendations_agent",
                END: END
            }
        )

        graph_builder.add_edge("flight_agent", END)
        graph_builder.add_edge("hotel_agent", END)
        graph_builder.add_edge("recommendations_agent", END)

        # 6. Set up memory (checkpointing) 👈 UPDATED STEP
        memory = MemorySaver()  # In-memory checkpointing for context

        # 7. Compile the graph, passing the memory object 👈 NEW STEP
        return  graph_builder.compile(checkpointer=memory)

    async def process_travel_request(self, user_request: str) -> str:
        """Process travel request using the compiled LangGraph supervisor."""
        logger.info(f"Processing request: {user_request}")
        
        try:
            # 1. Configuration object with thread_id for memory
            config = {"configurable": {"thread_id": self.thread_id}}

            # 2. Invoke the coordinator, passing the configuration
            response = await self.travel_coordinator.ainvoke(
                input={
                    "messages": [HumanMessage(content=user_request)]
                },
                config=config 
            )
            
            if response and "messages" in response:
                last_message = response["messages"][-1]
                response_text = getattr(last_message, 'content', str(last_message))
                
                logger.info("Agent response received")
                return response_text
            else:
                logger.warning("No final response received.")
                return "I apologize, but I wasn't able to process your travel request. Please try again."
                
        except Exception as e:
            logger.error(f"Error processing agent request: {e}")
            return f"An unexpected error occurred: {e}"




# Interactive terminal interface
if __name__ == "__main__":
    
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Please set the OPENAI_API_KEY environment variable.")
    else:
        # Pass the compiled graph to the class
        agent = LangChainTravelAgentDemo() 

        print("Welcome to the LangGraph Travel Agent Demo! Type 'exit' to quit.")
        print("Try a multi-turn conversation: 'Find me a flight to Delhi.' then 'What about a hotel there?'")

        while True:
            try:
                user_input = input("\nYour request: ").strip()
                if user_input.lower() in {"exit", "quit"}:
                    print("Goodbye!")
                    break
                
                result = asyncio.run(agent.process_travel_request(user_input))
                print(f"\nAgent: {result}")

            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")