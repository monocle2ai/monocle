import logging
import os
from pathlib import Path
from typing import Any, Dict

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up trace output path for test cases
tfwk_root = Path(__file__).parent.parent.parent  # Navigate to tfwk directory
trace_output_path = tfwk_root / ".monocle" / "test_traces"
os.environ["MONOCLE_TRACE_OUTPUT_PATH"] = str(trace_output_path)

# Set up Monocle telemetry for file logging
setup_monocle_telemetry(
    workflow_name="langchain_travel_agent_demo",
    monocle_exporters_list="file"
)

# Constants for session management
APP_NAME = "langchain_travel_agent_demo"
USER_ID = "travel_user_123"
SESSION_ID = "travel_session_456"


def book_flight(from_city: str, to_city: str, travel_date: str = "2025-12-01", is_business: bool = False) -> Dict[str, Any]:
    """Books a flight from one city to another.

    Args:
        from_city (str): The city from which the flight departs.
        to_city (str): The city to which the flight arrives.
        travel_date (str): The date of the flight in YYYY-MM-DD format.
        is_business (bool): Whether to book business class.

    Returns:
        Dict[str, Any]: Flight booking details with status, booking ID, and cost.
    """
    flight_class = "business" if is_business else "economy"
    base_price = 800 if is_business else 400
    
    # Add premium for popular destinations
    popular_destinations = ["Mumbai", "Delhi", "Goa", "Bangalore"]
    if to_city in popular_destinations:
        base_price += 200
    
    booking = {
        "status": "success",
        "booking_id": f"LC-FL-{hash(f'{from_city}-{to_city}') % 10000:04d}",
        "type": "flight",
        "from_city": from_city,
        "to_city": to_city,
        "travel_date": travel_date,
        "class": flight_class,
        "price": base_price,
        "message": f"Successfully booked {flight_class} flight from {from_city} to {to_city} on {travel_date}"
    }
    
    logger.info(f"LangChain Flight booked: {booking['booking_id']} - {from_city} to {to_city}")
    return booking


def book_hotel(hotel_name: str, city: str, check_in_date: str = "2025-12-01", nights: int = 1, is_business: bool = False) -> Dict[str, Any]:
    """Books a hotel for a stay.

    Args:
        hotel_name (str): The name of the hotel to book.
        city (str): The city where the hotel is located.
        check_in_date (str): The check-in date in YYYY-MM-DD format.
        nights (int): The number of nights to stay.
        is_business (bool): Whether this is a business booking.

    Returns:
        Dict[str, Any]: Hotel booking details with status, booking ID, and cost.
    """
    hotel_tier = "luxury" if is_business else "standard"
    base_price_per_night = 300 if is_business else 150
    total_price = base_price_per_night * nights
    
    booking = {
        "status": "success",
        "booking_id": f"LC-HT-{hash(f'{hotel_name}-{city}') % 10000:04d}",
        "type": "hotel",
        "hotel_name": hotel_name,
        "city": city,
        "check_in_date": check_in_date,
        "nights": nights,
        "tier": hotel_tier,
        "price_per_night": base_price_per_night,
        "total_price": total_price,
        "message": f"Successfully booked {nights} nights at {hotel_name} in {city} starting {check_in_date}"
    }
    
    logger.info(f"LangChain Hotel booked: {booking['booking_id']} - {hotel_name} in {city}")
    return booking


def get_travel_recommendations(destination: str) -> Dict[str, Any]:
    """Get travel recommendations for a destination.

    Args:
        destination (str): The destination city.

    Returns:
        Dict[str, Any]: Travel recommendations including attractions and tips.
    """
    recommendations = {
        "Mumbai": {
            "attractions": ["Gateway of India", "Marine Drive", "Bollywood Studios"],
            "best_time": "October to March",
            "tips": "Try local street food, book hotels near transportation hubs"
        },
        "Delhi": {
            "attractions": ["Red Fort", "India Gate", "Lotus Temple"],
            "best_time": "October to April",
            "tips": "Use metro for transportation, visit in early morning for less crowds"
        },
        "Goa": {
            "attractions": ["Beaches", "Old Goa Churches", "Spice Plantations"],
            "best_time": "November to March",
            "tips": "Rent a scooter, try seafood, book beach-side accommodations"
        }
    }
    
    result = {
        "destination": destination,
        "recommendations": recommendations.get(destination, {
            "attractions": ["Local markets", "Cultural sites", "Natural landmarks"],
            "best_time": "Check local weather patterns",
            "tips": "Research local customs and transportation options"
        }),
        "status": "success"
    }
    
    logger.info(f"LangChain Travel recommendations provided for: {destination}")
    return result


@tool
def book_flight_tool(from_city: str, to_city: str, travel_date: str = "2025-12-01", is_business: bool = False) -> str:
    """Book a flight between cities."""
    result = book_flight(from_city, to_city, travel_date, is_business)
    return result["message"]


@tool
def book_hotel_tool(hotel_name: str, city: str, check_in_date: str = "2025-12-01", nights: int = 1, is_business: bool = False) -> str:
    """Book a hotel reservation."""
    result = book_hotel(hotel_name, city, check_in_date, nights, is_business)
    return result["message"]


@tool
def get_recommendations_tool(destination: str) -> str:
    """Get travel recommendations for a destination."""
    result = get_travel_recommendations(destination)
    recommendations = result["recommendations"]
    return f"For {destination}: Attractions - {', '.join(recommendations['attractions'])}. Best time: {recommendations['best_time']}. Tips: {recommendations['tips']}"


class LangChainTravelAgentDemo:
    """LangChain Travel Agent demonstration with multi-agent architecture."""
    
    def __init__(self):
        # Store instance reference for delegation tools
        self._instance = None
        
        # Create specialized agents for different aspects of travel booking
        self.flight_agent = create_react_agent(
            model=ChatOpenAI(model="gpt-4o"),
            tools=[book_flight_tool],
            prompt=(
                "You are a helpful flight booking assistant. You can book flights between cities. "
                "Always confirm the travel details and provide flight booking information. "
                "Ask for clarification if travel dates or destinations are unclear."
            ),
            name="Flight_Assistant",
        )

        self.hotel_agent = create_react_agent(
            model=ChatOpenAI(model="gpt-4o"),
            tools=[book_hotel_tool],
            prompt=(
                "You are a helpful hotel booking assistant. You can book hotels in various cities. "
                "Always confirm accommodation preferences and provide booking details. "
                "Ask for clarification if check-in dates or hotel preferences are unclear."
            ),
            name="Hotel_Assistant",
        )

        self.recommendations_agent = create_react_agent(
            model=ChatOpenAI(model="gpt-4o"),
            tools=[get_recommendations_tool],
            prompt=(
                "You are a travel recommendations expert. Provide helpful travel advice, "
                "local attractions, best times to visit, and practical tips for destinations. "
                "Be informative and helpful in your recommendations."
            ),
            name="Recommendations_Assistant",
        )

        self.travel_supervisor = create_supervisor(
            agents=[self.flight_agent, self.hotel_agent, self.recommendations_agent],
            model=ChatOpenAI(model="gpt-4o"),
            prompt=(
                "You are a master travel planning agent that coordinates flight booking, hotel reservations, "
                "and travel recommendations to provide comprehensive travel planning services. "
                "Delegate flight bookings to the Flight Assistant, hotel bookings to the Hotel Assistant, "
                "and travel recommendations to the Recommendations Assistant."
            ),
            supervisor_name="Travel_Coordinator",
        ).compile()

    async def process_travel_request(self, user_request: str) -> str:
        """Process travel request using LangChain agents."""
        logger.info(f"LangChain Processing travel request: {user_request}")
        
        try:
            # Process the request through the supervisor agent using the correct API
            response = await self.travel_supervisor.ainvoke(
                input={
                    "messages": [
                        {
                            "role": "user",
                            "content": user_request,
                        }
                    ]
                }
            )
            
            # Extract response from LangChain agent response
            if response and "messages" in response:
                last_message = response["messages"][-1]
                if hasattr(last_message, 'content'):
                    response_text = last_message.content
                elif isinstance(last_message, dict) and 'content' in last_message:
                    response_text = last_message['content']
                else:
                    response_text = str(last_message)
                
                logger.info("LangChain Agent response received")
                return response_text
            else:
                logger.warning("No response received from LangChain agent")
                return "I apologize, but I wasn't able to process your travel request. Please try again."
                
        except Exception as e:
            logger.error(f"Error processing LangChain agent request: {e}")
            raise e



