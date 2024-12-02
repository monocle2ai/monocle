import os

from configs.agents import triage_agent
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from swarm.repl import run_demo_loop

from monocle_apptrace.exporters.file_exporter import UpdatedFileSpanExporter
from monocle_apptrace.instrumentor import setup_monocle_telemetry

context_variables = {
    "customer_context": """Here is what you know about the customer's details:
1. CUSTOMER_ID: customer_12345
2. NAME: John Doe
3. PHONE_NUMBER: (123) 456-7890
4. EMAIL: johndoe@example.com
5. STATUS: Premium
6. ACCOUNT_STATUS: Active
7. BALANCE: $0.00
8. LOCATION: 1234 Main St, San Francisco, CA 94123, USA
""",
    "flight_context": """The customer has an upcoming flight from LGA (Laguardia) in NYC to LAX in Los Angeles.
The flight # is 1919. The flight departure date is 3pm ET, 5/21/2024.""",
}


setup_monocle_telemetry(
    workflow_name="airline_flow",
    span_processors=[BatchSpanProcessor(UpdatedFileSpanExporter())],
    # wrapper_methods=[
    #             WrapperMethod(
    #                 package="swarm",
    #                 object_name="Swarm",
    #                 method="get_chat_completion",
    #                 span_name="jlt",
    #                 wrapper= llm_wrapper)
    #         ]
    )

if __name__ == "__main__":
    run_demo_loop(triage_agent, context_variables=context_variables)#debug=True
