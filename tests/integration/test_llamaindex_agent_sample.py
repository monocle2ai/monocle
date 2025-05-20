import time
from tests.common.custom_exporter import CustomConsoleSpanExporter
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from llama_index.core.agent import ReActAgent
import pytest
custom_exporter = CustomConsoleSpanExporter()
@pytest.fixture(scope="module")
def setup():
    setup_monocle_telemetry(
        workflow_name="llama_index_1",
        span_processors=[BatchSpanProcessor(custom_exporter)],
        wrapper_methods=[]
    )

# Define coffee menu
COFFEE_MENU = {
    "espresso": 2.5,
    "latte": 3.5,
    "cappuccino": 4.0,
    "americano": 3.0
}

# Define tools for the chatbot

def get_coffee_menu() -> str:
    """Return the available coffee menu."""
    menu_str = "\n".join([f"{item}: ${price:.2f}" for item, price in COFFEE_MENU.items()])
    return f"Available coffee options:\n{menu_str}"

coffee_menu_tool = FunctionTool.from_defaults(
    fn=get_coffee_menu,
    name="get_coffee_menu",
    description="Provides a list of available coffee options with prices."
)

def place_order(coffee_type: str, quantity: int) -> str:
    """Places an order for coffee."""
    if coffee_type.lower() not in COFFEE_MENU:
        return f"Sorry, {coffee_type} is not available. Please choose from the menu."
    total_cost = COFFEE_MENU[coffee_type.lower()] * quantity
    return f"Your order for {quantity} {coffee_type}(s) is confirmed. Total cost: ${total_cost:.2f}"

order_tool = FunctionTool.from_defaults(
    fn=place_order,
    name="place_order",
    description="Takes a coffee order and provides the total cost."
)

# Initialize LlamaIndex ReAct agent
llm = OpenAI(model="gpt-4")
agent = ReActAgent.from_tools([coffee_menu_tool, order_tool], llm=llm)

def test_llamaindex_agent(setup):
    print("Welcome to the Coffee Bot! ")
    user_input = "Please order 3 expresso coffees"
    response = agent.chat(user_input)
    time.sleep(5)
    print(f"Bot: {response}")

    spans = custom_exporter.get_captured_spans()
    for span in spans:
        span_attributes = span.attributes

        if "span.type" in span_attributes and (
            span_attributes["span.type"] == "inference" or span_attributes["span.type"] == "inference.framework"):
            # Assertions for all inference attributes
            assert span_attributes["entity.1.type"] == "inference.openai"
            assert "entity.1.provider_name" in span_attributes
            assert "entity.1.inference_endpoint" in span_attributes
            assert span_attributes["entity.2.name"] == "gpt-4"
            assert span_attributes["entity.2.type"] == "model.llm.gpt-4"

            # Assertions for metadata
            span_input, span_output, span_metadata = span.events
            assert "completion_tokens" in span_metadata.attributes
            assert "prompt_tokens" in span_metadata.attributes
            assert "total_tokens" in span_metadata.attributes

        if "span.type" in span_attributes and span_attributes["span.type"] == "agent":
            # Assertions for all inference attributes
            assert span_attributes["entity.1.name"] == "ReActAgent"
            assert span_attributes["entity.1.type"] == "Agent.oai"
            assert span_attributes["entity.1.tools"] == ("place_order",)


