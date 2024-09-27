# Monocle for tracing GenAI app code

**Monocle** helps developers and platform engineers building or managing GenAI apps monitor these in prod by making it easy to instrument their code to capture traces that are compliant with open-source cloud-native observability ecosystem. 

**Monocle** is a community-driven OSS framework for tracing GenAI app code governed as a [Linux Foundation AI & Data project](https://lfaidata.foundation/projects/monocle/). 

## Why Monocle

Monocle is built for: 
- **app developers** to trace their app code in any environment without writing code for every function 
- **platform engineers** to instrument apps in prod through wrapping instead of asking app devs to recode
- **GenAI component providers** to add observability features to their products 
- **enterprises** with existing open source observability stacks to consume traces from GenAI apps

Benefits:
- Monocle provides an implementation with packages that you can use right away, not just a specification
   - You don't have to learn the OpenTelemetry specification
   - You don't have to do bespoke implementation of that specification
   - You don't have to write lots of last-mile GenAI domain specific code to instrument your app
- Monocle provides consistency across components and environments 
   - You can connect traces across app code executions, model inference or data retrievals
   - You don't have to transform or cleanse telemetry data to add consistency across GenAI component providers
   - You use it in your personal lab development or organizational cloud production environments the same way
   - You can easily configure where the traces are sent to fit your scale, budget and observability stack
- Monocle is fully open source and community driven which means
   - no vendor lock-in
   - implementation is transparent
   - you can freely customize or add to it to fit your needs 

## What Monocle provides

- One-line code instrumentation 
- OpenTelemetry compatible format for traces
- Community-curated meta-model for tracing GenAI components
- Export to cloud storage  

## Use Monocle

- Get the Monocle package
  
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
  

## Roadmap 

Goal of Monocle is to support tracing for apps written in *any language* with *any LLM orchestration or agentic framework* and built using models, vectors, agents or other components served up by *any cloud or model inference provider*. 

Current version supports: 
- Language: (🟢) Python , (🔜) [Typescript](https://github.com/monocle2ai/monocle-typescript) 
- LLM-frameworks: (🟢) Langchain, (🟢) Llamaindex, (🟢) Haystack, (🔜) Flask
- LLM inference providers: (🟢) OpenAI, (🟢) Azure OpenAI, (🟢) Nvidia Triton, (🔜) AWS Bedrock, (🔜) Google Vertex, (🔜) Azure ML, (🔜) Hugging Face
- Vector stores: (🟢) FAISS, (🔜) OpenSearch, (🔜) Milvus
- Exporter: (🟢) stdout, (🟢) file, (🔜) Azure Blob Storage, (🔜) AWS S3, (🔜) Google Cloud Storage


## Get involved
### Provide feedback
- Submit issues and enhancements requests via Github issues

### Contribute
- Monocle is community based open source project. We welcome your contributions. Please refer to the CONTRIBUTING and CODE_OF_CONDUCT for guidelines. The [contributor's guide](CONTRIBUTING.md) provides technical details of the project.

