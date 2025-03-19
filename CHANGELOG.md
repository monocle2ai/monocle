## Version 0.3.0 (2024-12-10)

- Fixed issue with passing context in async case ([#150](https://github.com/monocle2ai/monocle/pull/150))
- Added lambda processor ([#148](https://github.com/monocle2ai/monocle/pull/148))
- Setup package level init scripts to make the monocle import simpler ([#147](https://github.com/monocle2ai/monocle/pull/147))
- Boto attributes and test cleanup ([#146](https://github.com/monocle2ai/monocle/pull/146))
- Openai workflow ([#142](https://github.com/monocle2ai/monocle/pull/142))
- Add input/output for openai embedding ([#141](https://github.com/monocle2ai/monocle/pull/141))
- Async method and scope fix ([#140](https://github.com/monocle2ai/monocle/pull/140))
- Bug fix for helper langchain and langgraph ([#137](https://github.com/monocle2ai/monocle/pull/137))
- Package main script to run any app with monocle instrumentation ([#132](https://github.com/monocle2ai/monocle/pull/132))
- Add openai api metamodel ([#131](https://github.com/monocle2ai/monocle/pull/131))
- Support notion of scopes to group traces/snaps into logical constructs ([#130](https://github.com/monocle2ai/monocle/pull/130))
- Add Llamaindex ReAct agent ([#127](https://github.com/monocle2ai/monocle/pull/127))
- Langhcain input fix and s3 exporter prefix support ([#126](https://github.com/monocle2ai/monocle/pull/126))
- Use standard AWS credential envs ([#123](https://github.com/monocle2ai/monocle/pull/123))
- Check additional attributes for Azure OpenAI model and consolidate common method in utils ([#122](https://github.com/monocle2ai/monocle/pull/122))
- Bug fix for accessor ([#121](https://github.com/monocle2ai/monocle/pull/121))
- Bug fix for empty response ([#120](https://github.com/monocle2ai/monocle/pull/120))
- Bug fix for inference endpoint ([#119](https://github.com/monocle2ai/monocle/pull/119))
- Opendal exporter for S3 and Blob ([#117](https://github.com/monocle2ai/monocle/pull/117))
- Handle specific ModuleNotFoundError exceptions gracefully ([#115](https://github.com/monocle2ai/monocle/pull/115))
- Adding support for console and memory exporter to list of monocle exporters ([#113](https://github.com/monocle2ai/monocle/pull/113))
- Add trace id propogation for constant trace id and from request ([#111](https://github.com/monocle2ai/monocle/pull/111))
- Restructure of monoocle code for easy extensibility ([#109](https://github.com/monocle2ai/monocle/pull/109))
- S3 update filename prefix ([#98](https://github.com/monocle2ai/monocle/pull/98))
- Update inference span for botocore sagemaker ([#93](https://github.com/monocle2ai/monocle/pull/93))
- Capturing inference output and token metadata for bedrock ([#82](https://github.com/monocle2ai/monocle/pull/82))
- Add dev dependency for Mistral AI integration ([#81](https://github.com/monocle2ai/monocle/pull/81))
- Add VectorStore deployment URL capture support ([#80](https://github.com/monocle2ai/monocle/pull/80))  
- Clean up cloud exporter implementation ([#79](https://github.com/monocle2ai/monocle/pull/79))
- Capture inference span input/output events attributes ([#77](https://github.com/monocle2ai/monocle/pull/77))
- Add release automation workflows ([#76](https://github.com/monocle2ai/monocle/pull/76))
- Fix gaps in Monocle SDK implementation ([#72](https://github.com/monocle2ai/monocle/pull/72))  
- Add kwargs and return value handling in Accessor ([#71](https://github.com/monocle2ai/monocle/pull/71))
- Update workflow name formatting ([#69](https://github.com/monocle2ai/monocle/pull/69))
- Implement Haystack metamodel support ([#68](https://github.com/monocle2ai/monocle/pull/68))

## Version 0.2.0 (2024-12-05)

## 0.2.0 (Oct 22, 2024)

- Ndjson format for S3 and Blob exporters ([#61](https://github.com/monocle2ai/monocle/pull/61))
- Set monocle exporter from env setting ([#60](https://github.com/monocle2ai/monocle/pull/60))
- Update workflow name and type with new format ([#59](https://github.com/monocle2ai/monocle/pull/59))
- Updated async and custom output processor testcase for metamodel([#58](https://github.com/monocle2ai/monocle/pull/58))
- Build okahu exporter and added test cases for okahu exporte ([#56](https://github.com/monocle2ai/monocle/pull/56))
- Handle exception in span wrappers([#52](https://github.com/monocle2ai/monocle/pull/52))
- Metamodel entity changes ([#51](https://github.com/monocle2ai/monocle/pull/51)), ([#54](https://github.com/monocle2ai/monocle/pull/54))
- Error handling for llm_endpoint and tags ([#50](https://github.com/monocle2ai/monocle/pull/50))
- Context_output for vector store retriever ([#48](https://github.com/monocle2ai/monocle/pull/48))
- Direct exporter - AWS S3 ([#42](https://github.com/monocle2ai/monocle/pull/42))
- Direct Exporter - Blob store ([#41](https://github.com/monocle2ai/monocle/pull/41))
- Initial metamodel definition ([#39](https://github.com/monocle2ai/monocle/pull/39))
- Improvement in vectorstore traces ([#38](https://github.com/monocle2ai/monocle/pull/38))
- Update key for session context field in attributes ([#34](https://github.com/monocle2ai/monocle/pull/34))


## 0.1.0 (Aug 27, 2024)

- Fixed LlamaIndex tracing bugs ([#32](https://github.com/monocle2ai/monocle/pull/32))
- Added support to add AWS cloud infra attributes ([#29](https://github.com/monocle2ai/monocle/pull/29))
- Added support to add Azure cloud infra attributes ([#23](https://github.com/monocle2ai/monocle/pull/23))
- Added support for adding provider name in LLM span in traces ([#22](https://github.com/monocle2ai/monocle/pull/22))
- Added a default file span exporter ([#21](https://github.com/monocle2ai/monocle/pull/21))
- Moved input and output context and prompts from attributes to events ([#15](https://github.com/monocle2ai/monocle/pull/15))






## 0.0.1 (Jul 17, 2024)

- First monocle release
