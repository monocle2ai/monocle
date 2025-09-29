# Ollama Instrumentation

This module provides OpenTelemetry instrumentation support for Ollama Python client library.

## Supported Operations

### Chat Completions
- `ollama.chat()` - Direct function call
- `ollama.Client.chat()` - Synchronous client method
- `ollama.AsyncClient.chat()` - Asynchronous client method
- Streaming support for all chat methods

### Text Generation
- `ollama.Client.generate()` - Synchronous text generation
- `ollama.AsyncClient.generate()` - Asynchronous text generation

### Embeddings
- `ollama.Client.embed()` - Synchronous embeddings
- `ollama.AsyncClient.embed()` - Asynchronous embeddings
- `ollama.Client.embeddings()` - Legacy embeddings method
- `ollama.AsyncClient.embeddings()` - Legacy async embeddings

## Instrumented Attributes

### Inference Spans
- `type`: "inference.ollama"
- `provider_name`: Ollama endpoint (e.g., "ollama.local")
- `inference_endpoint`: Full endpoint URL (e.g., "http://localhost:11434/")
- `model_name`: Model used for inference
- `input`: Request messages or prompt
- `response`: Generated response text
- `finish_reason`: Completion reason (if available)
- `finish_type`: Mapped finish type

### Retrieval Spans (Embeddings)
- `type`: "retrieval.ollama" 
- `provider_name`: Ollama endpoint
- `inference_endpoint`: Full endpoint URL
- `model_name`: Embedding model name
- `input`: Text to embed
- `response`: Generated embedding vector

## Usage

The instrumentation is automatically applied when you set up Monocle telemetry:

```python
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from ollama import chat

# Setup telemetry
setup_monocle_telemetry(workflow_name="my_ollama_app")

# Your Ollama calls will now be instrumented
response = chat(model='llama2', messages=[
    {'role': 'user', 'content': 'Hello!'}
])
```

## Stream Processing

The instrumentation supports Ollama's streaming responses and will:
- Capture time to first token
- Accumulate streaming content
- Record completion metadata
- Handle both sync and async streaming

## Error Handling

The instrumentation gracefully handles:
- Connection errors to Ollama server
- Invalid model names
- Malformed requests
- Streaming interruptions
