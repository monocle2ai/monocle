import logging
from typing import Any, Dict

from agents import Agent, Runner, function_tool

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants for session management
APP_NAME = "openai_travel_agent_demo"
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
        "booking_id": f"OAI-FL-{hash(f'{from_city}-{to_city}') % 10000:04d}",
        "type": "flight",
        "from_city": from_city,
        "to_city": to_city,
        "travel_date": travel_date,
        "class": flight_class,
        "price": base_price,
        "message": f"Successfully booked {flight_class} flight from {from_city} to {to_city} on {travel_date}"
    }
    
    logger.info(f"OpenAI Flight booked: {booking['booking_id']} - {from_city} to {to_city}")
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
        "booking_id": f"OAI-HT-{hash(f'{hotel_name}-{city}') % 10000:04d}",
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
    
    logger.info(f"OpenAI Hotel booked: {booking['booking_id']} - {hotel_name} in {city}")
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
    
    logger.info(f"OpenAI Travel recommendations provided for: {destination}")
    return result


class OpenAITravelAgentDemo:
    """OpenAI Agents SDK Travel Agent demonstration with proper session management."""
    
    def __init__(self):
        self.Runner = Runner
        
        # Create function tools
        @function_tool
        def book_flight_tool(from_city: str, to_city: str, travel_date: str = "2025-12-01", is_business: bool = False) -> str:
            """Book a flight between cities."""
            result = book_flight(from_city, to_city, travel_date, is_business)
            return result["message"]

        @function_tool
        def book_hotel_tool(hotel_name: str, city: str, check_in_date: str = "2025-12-01", nights: int = 1, is_business: bool = False) -> str:
            """Book a hotel reservation."""
            result = book_hotel(hotel_name, city, check_in_date, nights, is_business)
            return result["message"]

        @function_tool
        def get_recommendations_tool(destination: str) -> str:
            """Get travel recommendations for a destination."""
            result = get_travel_recommendations(destination)
            recommendations = result["recommendations"]
            return f"For {destination}: Attractions - {', '.join(recommendations['attractions'])}. Best time: {recommendations['best_time']}. Tips: {recommendations['tips']}"
        
        # Create specialized agents for different aspects of travel booking
        self.flight_agent = Agent(
            name="Flight Assistant",
            instructions=(
                "You are a helpful flight booking assistant. You can book flights between cities. "
                "Always confirm the travel details and provide flight booking information. "
                "Ask for clarification if travel dates or destinations are unclear."
            ),
            tools=[book_flight_tool]
        )

        self.hotel_agent = Agent(
            name="Hotel Assistant", 
            instructions=(
                "You are a helpful hotel booking assistant. You can book hotels in various cities. "
                "Always confirm accommodation preferences and provide booking details. "
                "Ask for clarification if check-in dates or hotel preferences are unclear."
            ),
            tools=[book_hotel_tool]
        )

        self.recommendations_agent = Agent(
            name="Recommendations Assistant",
            instructions=(
                "You are a travel recommendations expert. Provide helpful travel advice, "
                "local attractions, best times to visit, and practical tips for destinations. "
                "Be informative and helpful in your recommendations."
            ),
            tools=[get_recommendations_tool]
        )

        # Supervisor agent that coordinates all travel services
        self.travel_supervisor = Agent(
            name="Travel Coordinator",
            instructions=(
                "You are a master travel planning agent that coordinates flight booking, hotel reservations, "
                "and travel recommendations to provide comprehensive travel planning services. "
                "Delegate flight bookings to the Flight Assistant, hotel bookings to the Hotel Assistant, "
                "and travel recommendations to the Recommendations Assistant."
            ),
            handoffs=[self.flight_agent, self.hotel_agent, self.recommendations_agent],
            tools=[book_flight_tool, book_hotel_tool, get_recommendations_tool]
        )
        
    def _mock_process_request(self, user_request: str) -> str:
        """Mock implementation when OpenAI Agents SDK is not available."""
        logger.info(f"Mock OpenAI Processing travel request: {user_request}")
        
        # Simple mock responses based on request content
        if "flight" in user_request.lower():
            return "I'd be happy to help you book a flight. Based on your request, I've found several options and can proceed with booking a business class flight from Delhi to Mumbai for December 15th, 2025 for $800."
        elif "hotel" in user_request.lower():
            return "I can help you book hotel accommodation. Based on your preferences, I've found suitable hotels in your destination and can book you a room at the Luxury Resort in Goa for 3 nights starting December 20th."
        elif "recommend" in user_request.lower() or "attraction" in user_request.lower():
            return "Here are my travel recommendations for Delhi: Visit Red Fort, India Gate, and Lotus Temple. Best time to visit is October to April. Use metro for transportation and visit early morning for fewer crowds."
        else:
            return "I'm your travel planning assistant. I can help you book flights, reserve hotels, and provide travel recommendations for your destination. How can I assist you today?"
        
    async def process_travel_request(self, user_request: str) -> str:

            
        logger.info(f"OpenAI Processing travel request: {user_request}")
        
        try:
            # Process the request through the supervisor agent using the correct API
            response = await self.Runner.run(self.travel_supervisor, user_request)
            
            if response and hasattr(response, 'final_output'):
                response_text = response.final_output
                logger.info("OpenAI Agent response received")
                return response_text
            elif response and hasattr(response, 'messages') and response.messages:
                # Get the last message from the agent
                last_message = response.messages[-1]
                if hasattr(last_message, 'text'):
                    response_text = last_message.text
                elif hasattr(last_message, 'content'):
                    response_text = last_message.content
                else:
                    response_text = str(last_message)
                
                logger.info("OpenAI Agent response received")
                return response_text
            else:
                logger.warning("No response received from OpenAI agent")
                return "I apologize, but I wasn't able to process your travel request. Please try again."
                
        except Exception as e:
            logger.error(f"Error processing OpenAI agent request: {e}")
            return self._mock_process_request(user_request)


