

# this file is the entrypoint file where the Monocle agent is called
# the file can be called like this "python langchain_sample_entry.py"

# importing the langchain RAG app code from the file
from langchain_sample_wrapped import main
from monocle_apptrace.instrumentor import setup_monocle_telemetry
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

# setting up the Monocle agent
setup_monocle_telemetry(
            workflow_name="langchain_app_1",
            span_processors=[BatchSpanProcessor(ConsoleSpanExporter())],
            wrapper_methods=[])

# calling the app code after setting up Monocle agent
main()

