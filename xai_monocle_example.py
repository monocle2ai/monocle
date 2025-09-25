"""
Example script showing how to use Monocle with xAI SDK.

This script demonstrates how to instrument xAI SDK calls with Monocle
to generate observability traces.

To run this script:
1. Install xAI SDK: pip install xai-sdk
2. Set up your xAI API key in environment: export XAI_API_KEY="your_key_here" 
3. Run: python xai_monocle_example.py
"""

import sys
import os
from dotenv import load_dotenv

# Add monocle to path (adjust this to your monocle installation)
sys.path.insert(0, '/Users/anshul/work/monocle/monocle/src')

# Set up Monocle instrumentation BEFORE importing xAI SDK
from monocle_apptrace.instrumentation import setup_monocle_telemetry
from monocle_apptrace.exporters.file_exporter import FileSpanExporter
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

def main():
    # Load environment variables
    load_dotenv()
    
    # Set up Monocle telemetry - this must be done before importing xAI SDK
    print("Setting up Monocle telemetry...")
    instrumentor = setup_monocle_telemetry(
        workflow_name="xai_coffee_assistant",
        span_processors=[
            SimpleSpanProcessor(ConsoleSpanExporter()),
            SimpleSpanProcessor(FileSpanExporter())
        ]
    )

    try:
        # Import xAI SDK after setting up instrumentation
        from xai_sdk import Client
        from xai_sdk.chat import system, user
        
        print("Creating xAI client...")
        client = Client()
        
        print("Creating chat session...")
        chat = client.chat.create(
            model="grok-4-fast-non-reasoning",
            messages=[system("""You are a Coffee assistant. Only answer questions relating to coffee.
                If you don't know the answer, politely decline answering the question.""")]
        )

        # Interactive chat loop
        print("\n=== Coffee Assistant Chat (type 'exit' to quit) ===")
        while True:
            prompt = input("You: ")
            if prompt.lower() == "exit":
                break
            
            # This will be traced by Monocle
            chat.append(user(prompt))
            response = chat.sample()  # This method call will generate a span
            print(f"Grok: {response.content}")
            chat.append(response)
            
    except ImportError as e:
        print(f"xAI SDK not installed: {e}")
        print("Please install it with: pip install xai-sdk")
        print("And set your API key: export XAI_API_KEY='your_key_here'")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up instrumentation
        if instrumentor:
            instrumentor.uninstrument()
        print("\nMonocle instrumentation cleaned up.")

if __name__ == "__main__":
    main()