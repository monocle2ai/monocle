import logging
import time
from crewai import Agent, Crew, Task
from crewai.tools import BaseTool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

from monocle_apptrace.instrumentation import setup_monocle_telemetry
    # Setup monocle telemetry
setup_monocle_telemetry(workflow_name="crewai_travel_agent")

logging.basicConfig(level=logging.WARN)

# Hotel booking tool
class BookHotelTool(BaseTool):
    name: str = "crew_book_hotel"
    description: str = "Book a hotel reservation"

    def _run(self, hotel_name: str, city: str, check_in_date: str, nights: int = 1) -> dict:
        """Books a hotel for a stay.

        Args:
            hotel_name (str): The name of the hotel to book.
            city (str): The city where the hotel is located.
            check_in_date (str): The check-in date for the hotel stay.
            nights (int): The number of nights for the hotel stay.

        Returns:
            dict: status and message.
        """
        time.sleep(0.1)  # Simulate processing time
        return {
            "status": "success",
            "message": f"Successfully booked a stay at {hotel_name} in {city} from {check_in_date} for {nights} nights."
        }


# Flight booking tool
class BookFlightTool(BaseTool):
    name: str = "crew_book_flight"
    description: str = "Book a flight reservation"

    def _run(self, from_airport: str, to_airport: str, date: str = "next week") -> dict:
        """Books a flight from one airport to another.

        Args:
            from_airport (str): The airport from which the flight departs.
            to_airport (str): The airport to which the flight arrives.
            date (str): The date of the flight.

        Returns:
            dict: status and message.
        """
        time.sleep(0.1)  # Simulate processing time
        return {
            "status": "success",
            "message": f"Flight booked from {from_airport} to {to_airport} for {date}."
        }


# Create tools
hotel_tool = BookHotelTool()
flight_tool = BookFlightTool()

def create_agents():
    """Create CrewAI agents. Only call this when OpenAI API key is available."""
    # Create hotel booking agent
    hotel_booking_agent = Agent(
        role="CrewAI Hotel Booking Agent",
        goal="Book the best hotel accommodations for travelers",
        backstory="You are an expert hotel booking specialist with extensive knowledge of hotels worldwide. You only handle hotel booking requests.",
        tools=[hotel_tool],
        llm=ChatOpenAI(model="gpt-4o-mini"),
        verbose=False,
        allow_delegation=False,
        step_callback=None
    )

    # Create flight booking agent
    flight_booking_agent = Agent(
        role="CrewAI Flight Booking Agent", 
        goal="Book the best flight options for travelers",
        backstory="You are an expert flight booking specialist with access to all major airlines. You only handle flight booking requests.",
        tools=[flight_tool],
        llm=ChatOpenAI(model="gpt-4o-mini"),
        verbose=False,
        allow_delegation=False,
        step_callback=None
    )

    # Create supervisor agent - with access to all tools to avoid delegation issues
    supervisor_agent = Agent(
        role="CrewAI Travel Agent",
        goal="Coordinate complete travel bookings by directly using specialist tools",
        backstory="You are a travel supervisor who can directly book hotels and flights using available tools. You coordinate all travel arrangements.",
        tools=[hotel_tool, flight_tool],  # Give supervisor direct access to tools
        llm=ChatOpenAI(model="gpt-4o-mini"),
        verbose=False,
        allow_delegation=False,  # Disable delegation to avoid validation errors,
        step_callback=None
    )
    
    return hotel_booking_agent, flight_booking_agent, supervisor_agent

# Initialize agents only when imported by tests (not when run directly)
hotel_booking_agent = None
flight_booking_agent = None 
supervisor_agent = None


def create_crewai_travel_crew(travel_request: str):
    """Create a CrewAI crew for travel booking based on the request."""
    global hotel_booking_agent, flight_booking_agent, supervisor_agent
    
    # Initialize agents if not already done
    if hotel_booking_agent is None:
        hotel_booking_agent, flight_booking_agent, supervisor_agent = create_agents()
    
    # Create tasks based on the request
    tasks = []
    
    # Check if hotel booking is needed
    if any(keyword in travel_request.lower() for keyword in ['hotel', 'stay', 'accommodation', 'room']):
        hotel_task = Task(
            name="Hotel Booking Task",
            description=f"Extract hotel booking details from this request and book accordingly: {travel_request}",
            expected_output="Hotel booking confirmation with details",
            agent=hotel_booking_agent
        )
        tasks.append(hotel_task)
    
    # Check if flight booking is needed
    if any(keyword in travel_request.lower() for keyword in ['flight', 'fly', 'travel', 'airport']):
        flight_task = Task(
            name="Flight Booking Task",
            description=f"Extract flight booking details from this request and book accordingly: {travel_request}",
            expected_output="Flight booking confirmation with details",
            agent=flight_booking_agent
        )
        tasks.append(flight_task)
    
    # Add supervisor task to coordinate
    supervisor_task = Task(
        name="Travel Coordination Task",
        description=f"Coordinate and summarize the complete travel booking for: {travel_request}. Ensure all requested services are booked and provide a comprehensive summary.",
        expected_output="Complete travel booking summary with all confirmations",
        agent=supervisor_agent
    )
    tasks.append(supervisor_task)

    # Create crew
    crew = Crew(
        agents=[supervisor_agent, hotel_booking_agent, flight_booking_agent],
        tasks=tasks,
        verbose=True,
        process="sequential"
    )
    return crew


def execute_crewai_travel_request(travel_request: str):
    """Execute a travel request using CrewAI and return the result."""
    crew = create_crewai_travel_crew(travel_request)
    result = crew.kickoff(inputs={
        "travel_request": travel_request
    })
    return str(result)


def interactive_session():
    """Run an interactive travel booking session."""
    print("🚀 CrewAI Travel Agent")
    print("Type 'quit' to exit")
    
    while True:
        try:
            travel_request = input("\n🎯 Enter travel request: ").strip()
            
            if travel_request.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
                
            if travel_request:
                print("⏳ Processing...")
                result = execute_crewai_travel_request(travel_request)
                print(f"✅ {result}")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break


if __name__ == "__main__":
    interactive_session()
