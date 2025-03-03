# Custom Instrumentation with Monocle

This guide demonstrates how to use Monocle to instrument OpenAI and Vector DB interactions, collecting telemetry data to analyze and monitor their performance.

## Overview

The example includes the following components:

- **OpenAI Client (`openai_client.py`)**: A client for interacting with OpenAI's Chat API
- **Vector Database (`vector_db.py`)**: An in-memory vector database with OpenAI embeddings
- **Output Processors**: Configuration files that define how to extract and structure telemetry data
- **Example script**: Shows how to instrument and run the application

## Component Details

### OpenAI Client

`OpenAIClient` is a wrapper around the OpenAI API that provides methods for:

- Making chat completion requests via the `chat()` method
- Formatting messages for the API using `format_messages()`
- Handling API responses and errors

```python
# Initialize client
client = OpenAIClient()

# Format messages and send to OpenAI
messages = client.format_messages(
    system_prompts=["You are a helpful assistant."],
    user_prompts=["Tell me a joke about programming."]
)
response = client.chat(messages=messages, model="gpt-3.5-turbo")
```

### Vector Database

`InMemoryVectorDB` is a simple vector database implementation that:

- Converts text to vector embeddings using OpenAI's embedding API
- Stores vectors with associated metadata
- Performs similarity searches using cosine similarity

```python
# Initialize vector database
vector_db = InMemoryVectorDB()

# Store documents
vector_db.store_text("doc1", "Python is a programming language", {"source": "docs"})

# Search for similar documents
results = vector_db.search_by_text("programming languages", top_k=2)
```

## Instrumenting Your Code with Monocle

### 1. Define Output Processors

Output processors define what data to extract from your methods. Two examples are provided:

#### Inference Output Processor

`output_processor_inference.py` defines how to extract data from OpenAI chat completions:

```python
INFERENCE_OUTPUT_PROCESSOR = {
    "type": "inference",
    "attributes": [
        [
            # Entity attributes for the provider
            {
                "attribute": "type",
                "accessor": lambda arguments: "openai"
            },
            {
                "attribute": "deployment",
                "accessor": lambda arguments: arguments['kwargs'].get('model', 'unknown')
            },
            # More attributes...
        ]
    ],
    "events": [
        {
            "name": "data.input",
            "attributes": [
                {
                    "attribute": "input",
                    "accessor": lambda arguments: [
                        msg["content"] 
                        for msg in arguments['kwargs'].get('messages', [])
                    ] if isinstance(arguments['kwargs'].get('messages'), list) else []
                }
            ]
        },
        # More events...
    ]
}
```

#### Vector DB Output Processor

`output_processor_vector.py` defines how to extract data from vector database operations:

```python
VECTOR_OUTPUT_PROCESSOR = {
    "type": "retrieval",
    "attributes": [
        [
            # Vector store attributes
            {
                "attribute": "name",
                "accessor": lambda arguments: type(arguments["instance"]).__name__,
            },
            # More attributes...
        ]
    ],
    "events": [
        {
            "name": "data.input",
            "attributes": [
                {
                    "attribute": "input",
                    "accessor": lambda arguments: arguments["args"][0] if arguments["args"] else None
                }
            ]
        },
        # More events...
    ]
}
```

### 2. Accessor Functions

The key to instrumentation is the `accessor` function, which extracts data from method calls:

- `arguments["instance"]`: The object instance (e.g., the OpenAIClient or InMemoryVectorDB)
- `arguments["args"]`: Positional arguments passed to the method
- `arguments["kwargs"]`: Keyword arguments passed to the method
- `arguments["result"]`: The return value from the method call

These give you access to all inputs, outputs, and context of the instrumented methods.

### 3. Configure Instrumentation

Set up Monocle's telemetry system with your output processors:

```python
from monocle_apptrace.instrumentation.common.wrapper_method import WrapperMethod
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry

setup_monocle_telemetry(
    workflow_name="openai.app",
    wrapper_methods=[
        WrapperMethod(
            package="openai_client",           # Module name
            object_name="OpenAIClient",        # Class name
            method="chat",                     # Method to instrument
            span_name="openai_client.chat",    # Span name in telemetry
            output_processor=INFERENCE_OUTPUT_PROCESSOR
        ),
        # More method wrappers...
    ]
)
```

## Running the Example

1. Ensure you have your **OpenAI API key** available:

```bash
export OPENAI_API_KEY=your_api_key_here
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Run the example script:

```bash
python example.py
# Or use the provided shell script
./run_example.sh
```

## Understanding the Telemetry Output

Monocle generates JSON trace files in your directory with names like:
`monocle_trace_openai.app_<trace_id>_<timestamp>.json`

### Output Format

The trace files contain structured telemetry data:

```json
{
    "name": "openai_client.chat",
    "context": { /* trace context */ },
    "attributes": {
        "entity.2.type": "openai",
        "entity.2.provider_name": "OpenAI",
        "entity.2.deployment": "gpt-3.5-turbo",
        "entity.2.inference_endpoint": "https://api.openai.com/v1",
        "entity.3.name": "gpt-3.5-turbo",
        "entity.3.type": "model.llm.gpt-3.5-turbo"
    },
    "events": [
        {
            "name": "data.input",
            "timestamp": "2025-02-27T10:36:49.985586Z",
            "attributes": {
                "input": [
                    "You are a helpful AI assistant.",
                    "Tell me a short joke about programming."
                ]
            }
        },
        {
            "name": "data.output",
            "attributes": {
                "response": "Why do programmers prefer dark mode? Because the light attracts bugs!"
            }
        },
        {
            "name": "metadata",
            "attributes": {
                "prompt_tokens": 26,
                "completion_tokens": 14,
                "total_tokens": 40
            }
        }
    ]
}
```

### Key Elements

1. **Attributes**: Contains information about the instrumented entity:
   - Model name and type
   - Deployment details
   - API endpoints
   - Provider information

2. **Events**: Contains captured events during the method execution:
   - `data.input`: The inputs provided to the method
   - `data.output`: The response or results from the method
   - `metadata`: Additional information like token usage

## Customizing for Your Application

To instrument your own code:

1. Create output processors tailored to your methods
2. Use accessor functions to extract the data you need
3. Set up telemetry with your method wrappers
4. Run your application and analyze the generated traces

By customizing the output processors, you can collect exactly the telemetry data you need from any Python method.
