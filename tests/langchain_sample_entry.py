# Copyright (C) Okahu Inc 2023-2024. All rights reserved

# this file is the entrypoint file where the Okahu agent is called
# the file can be called like this "python langchain_sample_entry.py"

# importing the langchain RAG app code from the file
from langchain_sample_wrapped import main
from okahu_apptrace.instrumentor import setup_okahu_telemetry
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

# setting up the okahu agent
# make sure the OKAHU_API_KEY and OKAHU_INGESTION_ENDPOINT environment variables are configured already
setup_okahu_telemetry(
            workflow_name="langchain_app_1",
            span_processors=[BatchSpanProcessor(ConsoleSpanExporter())],
            wrapper_methods=[])

# calling the app code after setting up Okahu agent
main()

