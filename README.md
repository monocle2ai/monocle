# Monocle for tracing GenAI app code

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

