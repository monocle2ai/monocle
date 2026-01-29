# Monocle for tracing GenAI app code

<h4 align="center">
    <a href="https://pypi.org/project/monocle-apptrace/" target="_blank">
        <img src="https://img.shields.io/pypi/v/monocle-apptrace.svg" alt="PyPI Version">
    </a>
    <a href="https://discord.gg/D8vDbSUhJX">
        <img src="https://img.shields.io/static/v1?label=Chat%20on&message=Discord&color=blue&logo=Discord&style=flat-square" alt="Discord">
    </a>
    <a href="https://join.slack.com/t/monocle2ai/shared_invite/zt-37pgez3jr-BNjNynF6VV8iHvRlaLM7QA">
        <img src="https://img.shields.io/static/v1?label=Chat%20on&message=Slack&color=black&logo=Slack&style=flat-square" alt="Slack">
    </a>
</h4>

**Monocle** helps developers and platform engineers building or managing GenAI apps monitor these in prod by making it easy to instrument their code to capture traces that are compliant with open-source cloud-native observability ecosystem. 

**Monocle** is a community-driven OSS framework for tracing GenAI app code governed as a [Linux Foundation AI & Data project](https://lfaidata.foundation/projects/monocle/). 

## Why Monocle

Monocle is built for: 
- **app developers** to trace their app code in any environment without lots of custom code decoration 
- **platform engineers** to instrument apps in prod through wrapping instead of asking app devs to recode
- **GenAI component providers** to add observability features to their products 
- **enterprises** to consume traces from GenAI apps in their existing open-source observability stack

Benefits:
- Monocle provides an implementation + package, not just a spec 
   - No expertise in OpenTelemetry spec required
   - No bespoke implementation of that spec required
   - No last-mile GenAI domain specific code required to instrument your app
- Monocle provides consistency  
   - Connect traces across app code executions, model inference or data retrievals
   - No cleansing of telemetry data across GenAI component providers required
   - Works the same in personal lab dev or org cloud prod environments
   - Send traces to location that fits your scale, budget and observability stack
- Monocle is fully open source and community driven
   - No vendor lock-in
   - Implementation is transparent
   - You can freely use or customize it to fit your needs 

## What Monocle provides

- Easy to [use](#use-monocle) code instrumentation
- OpenTelemetry compatible format for [spans](src/monocle_apptrace/metamodel/spans/span_format.json). 
- Community-curated and extensible [metamodel](src/monocle_apptrace/metamodel/README.md) for consisent tracing of GenAI components. 
- Export to local and cloud storage 

## Use Monocle

### Generate traces
Install monocle package and make simple two line change to your application to generate traces.
  
```
    pip install monocle_apptrace 
```
- Instrument your app code
     - Import the Monocle package
       ```
          from monocle_apptrace.instrumentor import setup_monocle_telemetry
       ```
     - Setup instrumentation in your ```main()``` function  
       ``` 
          setup_monocle_telemetry(workflow_name="your-app-name")
       ```         
- (Optionally) Modify config to alter where traces are sent

See [Monocle user guide](Monocle_User_Guide.md) for more details.
  
### Monocle test tool
Test your application and agent behavior by using Monocle's genAI test tool. Just define your input, expected output and expected agent or tool to be called. The Monocle test tool automatically generates the traces and validate your expected behavior.

### Monocle MCP server
The MCP server provided by Monocle integrates with your dev enviorment like Visual Studio and Github copilot. It provides curated prompts and tools to analyze the trace to find issues.

## Roadmap 

Goal of Monocle is to support tracing for apps written in *any language* with *any LLM orchestration or agentic framework* and built using models, vectors, agents or other components served up by *any cloud or model inference provider*. 

Current version supports: 
- Language: (🟢) Python , (🟢) [Typescript](https://github.com/monocle2ai/monocle-typescript)
- Agentic frameworks: (🟢) Langgraph, (🟢) LlamaIndex, (🟢) Google ADK, (🟢)  OpenAI Agent SDK, (🟢) AWS Strands, (🟢) CrewAI, (🟢) Microsoft Agent Framework
- MCP/A2A frameworks: (🟢) FastMCP, (🟢) MCP client, (🟢) A2A client
- Web/App frameworks: (🟢) Flask, (🟢) AIO Http, (🟢)FastAPI, (🟢) Azure Function, (🟢) AWS Lambda, (🟢) Vercel (typescript), (🟢) Microsoft Teams AI SDK, (🟢) Web/REST client, (🔜) Google Function, 
- LLM-frameworks: (🟢) Langchain, (🟢) Llamaindex, (🟢) Haystack
- Agent Runtime: (🟢) AWS Bedrock Agentcore
- LLM inference providers: (🟢) OpenAI, (🟢) Azure OpenAI, (🟢) Azure AI, (🟢) Nvidia Triton, (🟢) AWS Bedrock, (🟢) AWS Sagemaker, (🟢) Google Vertex, (🟢) Google Gemini, (🟢) Hugging Face, (🟢) Deepseek, (🟢) Anthropic, (🟢) Mistral, (🟢) LiteLLM ,(🔜) Azure ML
- Vector stores: (🟢) FAISS, (🔜) OpenSearch, (🔜) Milvus
- Exporter: (🟢) stdout, (🟢) file, (🟢) Memory, (🟢) Azure Blob Storage, (🟢) AWS S3, (🟢) Okahu cloud, (🟢) OTEL compatible collectors, (🔜) Google Cloud Storage

## Get involved
### Provide feedback
- Submit issues and enhancements requests via Github issues

### Contribute
- Monocle is community based open source project. We welcome your contributions. Please refer to the CONTRIBUTING and CODE_OF_CONDUCT for guidelines. The [contributor's guide](CONTRIBUTING.md) provides technical details of the project.

