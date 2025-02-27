from output_processor_inference import INFERENCE_OUTPUT_PROCESSOR
from output_processor_vector import VECTOR_OUTPUT_PROCESSOR
from monocle_apptrace.instrumentation.common.wrapper_method import WrapperMethod
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry

setup_monocle_telemetry(
    workflow_name="openai.app",
    wrapper_methods=[
        WrapperMethod(
            package="openai_client",
            object_name="OpenAIClient",
            method="chat",
            span_name="openai_client.chat",
            output_processor=INFERENCE_OUTPUT_PROCESSOR
        ),
        WrapperMethod(
            package="vector_db",
            object_name="InMemoryVectorDB",
            method="search_by_text",
            span_name="vector_db.search_by_text",
            output_processor=VECTOR_OUTPUT_PROCESSOR
        )
    ],
)

from openai_client import OpenAIClient
from vector_db import InMemoryVectorDB



def main():
    # Initialize clients
    client = OpenAIClient()
    vector_db = InMemoryVectorDB()  # Replace with your API key

    # Store some example texts
    vector_db.store_text(
        "doc1", 
        "Python is a high-level programming language",
        {"source": "programming-docs"}
    )
    vector_db.store_text(
        "doc2", 
        "Machine learning is a subset of artificial intelligence",
        {"source": "ml-docs"}
    )
    
    # Search using text query
    results = vector_db.search_by_text("programming languages", top_k=2)
    print("\nVector search results:")
    for result in results:
        print(f"ID: {result['id']}, Similarity: {result['similarity']:.3f}")
        print(f"Text: {result['metadata'].get('text', '')}")
    print(", ".join([result['metadata'].get('text', '') for result in results]))
    # Original OpenAI example
    system_prompts = ["You are a helpful AI assistant."]
    user_prompts = ["Tell me a short joke about programming."]
    messages = client.format_messages(system_prompts, user_prompts)
    response = client.chat(messages=messages, model="gpt-3.5-turbo", temperature=0.7)
    print("\nOpenAI response:")
    print(response["choices"][0]["message"]["content"])

if __name__ == "__main__":
    main()
