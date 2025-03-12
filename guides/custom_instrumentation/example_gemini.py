import os
import google.generativeai as genai
from output_processor_gemini import GEMINI_OUTPUT_PROCESSOR
from monocle_apptrace.instrumentation.common.wrapper_method import WrapperMethod
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry

# Set up Monocle telemetry with instrumentation for Gemini
setup_monocle_telemetry(
    workflow_name="gemini.app",
    wrapper_methods=[
        WrapperMethod(
            package="google.generativeai.generative_models",
            object_name="GenerativeModel",
            method="generate_content",
            span_name="gemini.generate_content",
            output_processor=GEMINI_OUTPUT_PROCESSOR
        )
    ],
)

def setup_gemini():
    """Set up the Gemini API with API key."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Google API key is required. Please set GOOGLE_API_KEY environment variable.")
    
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-1.5-flash')

def main():
    """Run examples using the Gemini API."""
    try:
        # Initialize Gemini model
        model = setup_gemini()
        
        # Example 1: Simple question
        prompt = "Explain quantum computing in simple terms in 3 lines."
        print(f"\nSending prompt to Gemini: {prompt}")
        
        response = model.generate_content(contents=prompt)
        
        print("\nGemini response:")
        print(response.text)
        
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    main()
