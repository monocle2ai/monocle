import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

# Assuming 'Agent' and 'Runner' are from the Agents SDK/LangGraph as implied
# If you are using Google's Agent SDK, the imports would look different.
# I'll keep the imports as you provided them for consistency.
from agents import Agent, Runner, function_tool
from mem0 import Memory  # Import the Mem0 library

logging.getLogger("httpx").setLevel(logging.WARNING)
# logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants for session management
APP_NAME = "openai_travel_agent_demo"
USER_ID = "travel_user_123" 
SESSION_ID = "travel_session_456" 


# --- Function Tools (Unchanged) ---

def book_flight(from_city: str, to_city: str, travel_date: str = "2025-12-01", is_business: bool = False) -> Dict[str, Any]:
    """Books a flight from one city to another."""
    flight_class = "business" if is_business else "economy"
    base_price = 800 if is_business else 400
    
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
    """Books a hotel for a stay."""
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
    """Get travel recommendations for a destination."""
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
    """
    OpenAI Agents SDK Travel Agent demonstration with EXPLICIT local mem0 session management.
    """
    
    def __init__(self):
        self.Runner = Runner
        
        # 1. Initialize a list to hold short-term chat history
        self.chat_history: List[Dict[str, str]] = []

        # Define an absolute path for better reliability
        chroma_path = Path.cwd() / "chroma_memories"

        # 🚀 FIX 1: The Configuration is now correct. Removed 'embedding_model_dims'.
        local_config = {
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "path": str(chroma_path),  # Chroma uses a directory for persistence
                    "collection_name": "travel_bookings"
                },
            },
        }
        
        try:
            self.memory = Memory.from_config(local_config)
            logger.info("Mem0 client initialized with LOCAL file-based storage.")
        except Exception as e:
            logger.error(f"Failed to initialize Mem0 locally. Ensure OPENAI_API_KEY is set. Error: {e}")
            self.memory = None 

        # --- Function Tools (Mem0 Add Calls Corrected) ---
        @function_tool
        def book_flight_tool(from_city: str, to_city: str, travel_date: str = "2025-12-01", is_business: bool = False) -> str:
            result = book_flight(from_city, to_city, travel_date, is_business)
            if self.memory:
                # ⬇️ .add() correctly uses user_id and session_id as top-level args
                memory_content = f"Fact: Flight Booking. {result['message']} - Booking ID: {result['booking_id']}. Details: {json.dumps(result)}"
                self.memory.add(
                    memory_content,
                    user_id=USER_ID,
                    run_id=SESSION_ID,
                    metadata={"type": "fact_booking_flight", "booking_id": result['booking_id'], "travel_date": travel_date}
                )
                logger.info(f"New flight booking stored in mem0 for user {USER_ID}.")
            return result["message"]

        @function_tool
        def book_hotel_tool(hotel_name: str, city: str, check_in_date: str = "2025-12-01", nights: int = 1, is_business: bool = False) -> str:
            result = book_hotel(hotel_name, city, check_in_date, nights, is_business)
            if self.memory:
                 # ⬇️ .add() correctly uses user_id and session_id as top-level args
                memory_content = f"Fact: Hotel Booking. {result['message']} - Booking ID: {result['booking_id']}. Details: {json.dumps(result)}"
                self.memory.add(
                    memory_content,
                    user_id=USER_ID,
                    run_id=SESSION_ID,
                    metadata={"type": "fact_booking_hotel", "booking_id": result['booking_id'], "check_in_date": check_in_date}
                )
                logger.info(f"New hotel booking stored in mem0 for user {USER_ID}.")
            return result["message"]

        @function_tool
        def get_recommendations_tool(destination: str) -> str:
            result = get_travel_recommendations(destination)
            recommendations = result["recommendations"]
            return f"For {destination}: Attractions - {', '.join(recommendations['attractions'])}. Best time: {recommendations['best_time']}. Tips: {recommendations['tips']}"
        
        # --- Agents Setup (Unchanged) ---
        self.flight_agent = Agent(
            name="Flight Assistant",
            instructions=("You are a helpful flight booking assistant..."), 
            tools=[book_flight_tool]
        )
        self.hotel_agent = Agent(
            name="Hotel Assistant", 
            instructions=("You are a helpful hotel booking assistant..."), 
            tools=[book_hotel_tool]
        )
        self.recommendations_agent = Agent(
            name="Recommendations Assistant",
            instructions=("You are a travel recommendations expert..."), 
            tools=[get_recommendations_tool]
        )

        self.travel_supervisor = Agent(
            name="Travel Coordinator",
            instructions=(
                "You are a master travel planning agent. You will be given context in two parts: "
                "'Prior Conversation History' (for conversational flow) and 'Retrieved Facts' (for past bookings)."
                "Your job is to respond to the 'User Request' based on this context."
                "\n\n"
                "**Your rules are:**"
                "\n"
                "1.  **If the user asks a question** (e.g., 'what are my bookings?', 'when is my flight?', 'retrieve my hotel info'), "
                "    you MUST answer the question **yourself** using the information from the 'Retrieved Facts' context. Do NOT delegate questions."
                "\n"
                "2.  **If the user asks to *book* a new flight**, delegate to the 'Flight Assistant'."
                "\n"
                "3.  **If the user asks to *book* a new hotel**, delegate to the 'Hotel Assistant'."
                "\n"
                "4.  **If the user asks for recommendations**, delegate to the 'Recommendations Assistant'."
            ),
            handoffs=[self.flight_agent, self.hotel_agent, self.recommendations_agent],
        )
        
    def _mock_process_request(self, user_request: str) -> str:
        return "Mock response."
        
    # --------------------------------------------------------------------------
    # ⬇️ MODIFIED: process_travel_request with the final Mem0.search() fix
    # --------------------------------------------------------------------------
    async def process_travel_request(self, user_request: str) -> str:

        logger.info(f"OpenAI Processing travel request: {user_request}")

        # --- STEP 4.1: Build context from Short-Term Memory (Chat History) ---
        short_term_memory_str = "\n".join([f"{turn['role']}: {turn['content']}" for turn in self.chat_history])

        # --- STEP 4.2: Build queries for Long-Term Memory (Mem0) ---
                
        # Use the dynamic, rich query
        mem0_search_query = f"""
        Prior Conversation:
        {short_term_memory_str}
        Current User Request: {user_request}

        Search for all user booking details, facts, flight bookings, and hotel reservations.
        """

        long_term_memory_str = ""

        if self.memory:
            # Always provide at least user_id and run_id as top-level args if required by Mem0
            try:
                relevant_memories = self.memory.search(
                    query=mem0_search_query,
                    user_id=USER_ID,
                    run_id=SESSION_ID
                )
            except TypeError:
                # Fallback: try with filters dict if Mem0 expects it that way
                filters = {
                    "user_id": USER_ID,
                    "run_id": SESSION_ID
                }
                relevant_memories = self.memory.search(
                    query=mem0_search_query,
                    filters=filters
                )

            if relevant_memories and relevant_memories.get('results'):
                # Use a set to store unique memories
                unique_memories = set()
                memories_list = []
                for m in relevant_memories['results']:
                    # We want memories that are facts/bookings AND not duplicate
                    if m.get('metadata', {}).get('type', '').startswith('fact_booking'):
                        if m['memory'] not in unique_memories:
                            memories_list.append(f"- {m['memory']}")
                            unique_memories.add(m['memory'])
                long_term_memory_str = "\n".join(memories_list)
                logger.info(f"Found {len(memories_list)} unique facts from Mem0.")
            else:
                logger.info("No relevant facts found in Mem0.")
        
        # --- STEP 4.3: Combine all context for the Agent ---
        contextual_user_request = f"""
        Here is the relevant context for this turn.

        --- Prior Conversation History (Short-Term) ---
        {short_term_memory_str if short_term_memory_str else "No prior conversation in this session."}

        --- Retrieved Facts (Long-Term from Mem0) ---
        {long_term_memory_str if long_term_memory_str else "No specific facts retrieved from long-term memory."}
        
        -------------------------------------------------

        User Request: {user_request}

        Now, fulfill the user's request based on *all* the context provided.
        """
        
        try:
            # Process the request through the supervisor agent
            response = await self.Runner.run(self.travel_supervisor, contextual_user_request)
            
            response_text = ""
            if response and hasattr(response, 'final_output'):
                response_text = response.final_output
            elif response and hasattr(response, 'messages') and response.messages:
                last_message = response.messages[-1]
                response_text = getattr(last_message, 'text', getattr(last_message, 'content', str(last_message)))
            else:
                response_text = "I apologize, but I wasn't able to process your travel request. Please try again."
            
            logger.info("OpenAI Agent response received")
            
            # --- STEP 4.4: Update Memories ---
            
            # ✍️ Update Long-Term Memory (Mem0) with this turn's summary
            if self.memory:
                self.memory.add(
                    f"User asked: {user_request}. Agent responded: {response_text}",
                    user_id=USER_ID,
                    run_id=SESSION_ID,
                    metadata={"type": "conversation_turn"}
                )
            
            # 📌 Update Short-Term Memory (Local)
            self.chat_history.append({"role": "user", "content": user_request})
            self.chat_history.append({"role": "assistant", "content": response_text})

            return response_text
                
        except Exception as e:
            logger.error(f"Error processing OpenAI agent request: {e}")
            return self._mock_process_request(user_request)


# Interactive terminal interface
if __name__ == "__main__":
    
    # Check for both necessary keys, as Mem0 OSS often uses OpenAI for embeddings
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Please set the OPENAI_API_KEY environment variable.")
    else:
        # NOTE: This part assumes the 'agents' module (Agent, Runner, function_tool) 
        # is available and correctly configured in your environment.
        agent = OpenAITravelAgentDemo() 

        print("\n" + "="*50)
        print("Welcome to the LangGraph Travel Agent Demo! Type 'exit' to quit.")
        print("💡 Try: 'Find me a flight to Delhi.' then 'What about a hotel there?'")
        print("💡 Then try: 'What flight did I book?' to test long-term memory.")
        print("="*50)

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