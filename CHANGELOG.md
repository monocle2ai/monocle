## Version 0.5.3 (2025-09-09)

- Make ./.monocle as default folder to store traces instead of current directory ([#272](https://github.com/monocle2ai/monocle/pull/272))
- HTTP span formatting ([#258](https://github.com/monocle2ai/monocle/pull/258))
- Adding mistral metamodel - instrumentation of inference api ([#273](https://github.com/monocle2ai/monocle/pull/273))
- Add Http Url for Azfunc, Lambda, Flask and FastApi Metamodel ([#254](https://github.com/monocle2ai/monocle/pull/254))
- Fixed input tool ([#271](https://github.com/monocle2ai/monocle/pull/271))
- Fix ([#270](https://github.com/monocle2ai/monocle/pull/270))

## Version 0.5.2 (2025-08-27)

- Add monocle MCP server ([#264](https://github.com/monocle2ai/monocle/pull/264))

## Version 0.5.1 (2025-08-25)

- Support injecting synthetic spans and use that for adding delegation span in Google ADK ([#257](https://github.com/monocle2ai/monocle/pull/257))
- Added monocle subtypes ([#256](https://github.com/monocle2ai/monocle/pull/256))
- OpenAI agents request span ([#255](https://github.com/monocle2ai/monocle/pull/255))
- Set scopes for agent request and invocation plus minor fixes ([#253](https://github.com/monocle2ai/monocle/pull/253))

## Version 0.5.0 (2025-08-05)

- Fix missing inference span in google ADK ([#249](https://github.com/monocle2ai/monocle/pull/249))
- Added openai agents instrumentation ([#248](https://github.com/monocle2ai/monocle/pull/248))
- Google Agent development kit meta mode ([#247](https://github.com/monocle2ai/monocle/pull/247))
- Add Support for LiteLLM OpenAI and AzureOpenAI ([#246](https://github.com/monocle2ai/monocle/pull/246))
- Added inference subtype for langgrap, openai and anthropic ([#245](https://github.com/monocle2ai/monocle/pull/245))
- Add support for teams finish type, move finish types to metadata section ([#240](https://github.com/monocle2ai/monocle/pull/240))
- Fix integration tests ([#239](https://github.com/monocle2ai/monocle/pull/239))
- Add function_name attribute for Azure function ([#238](https://github.com/monocle2ai/monocle/pull/238))
- Added MCP and A2A ([#237](https://github.com/monocle2ai/monocle/pull/237))
- Add Metamodel for Lambda Func ([#236](https://github.com/monocle2ai/monocle/pull/236))
- Allow llm SDKs as workflow types ([#234](https://github.com/monocle2ai/monocle/pull/234))
- Add Sample Span for VertexAI ([#233](https://github.com/monocle2ai/monocle/pull/233))
- Add Haystack Gemini sample ([#232](https://github.com/monocle2ai/monocle/pull/232))
- Add Llama Index Gemini sample ([#231](https://github.com/monocle2ai/monocle/pull/231))
- Add workflow span name to all spans ([#230](https://github.com/monocle2ai/monocle/pull/230))
- Add Langchain Gemini sample ([#229](https://github.com/monocle2ai/monocle/pull/229))
- Add Embeddings Span in Gemini Metamodel ([#228](https://github.com/monocle2ai/monocle/pull/228))
- Agent metamodel updates for LangGraph ([#227](https://github.com/monocle2ai/monocle/pull/227))
- Add fastapi metamodel with fastapi tracid propogation testcase ([#226](https://github.com/monocle2ai/monocle/pull/226))

## Version 0.4.2 (2025-06-26)

- Add gemini instrumentation ([#220](https://github.com/monocle2ai/monocle/pull/220))

## Version 0.4.1 (2025-06-17)

- Add exception status code for Boto3 ([#211](https://github.com/monocle2ai/monocle/pull/211))
- Add exception status code for anthropic and openai ([#210](https://github.com/monocle2ai/monocle/pull/210))
- Add prompt template info in teamsAI ([#209](https://github.com/monocle2ai/monocle/pull/209))
- TeamsAI : added system prompt ([#208](https://github.com/monocle2ai/monocle/pull/208))
- Add prompt template info in ActionPlanner for teamsAI ([#207](https://github.com/monocle2ai/monocle/pull/207))
- Add teams channel id as scope in MS teams instrumentations ([#206](https://github.com/monocle2ai/monocle/pull/206))
- Azure function wrapper to generate http span ([#205](https://github.com/monocle2ai/monocle/pull/205))
- Azure ai inference sdk ([#204](https://github.com/monocle2ai/monocle/pull/204))

## Version 0.4.0 (2025-06-02)

- Update teams scopes ([#200](https://github.com/monocle2ai/monocle/pull/200))
- Record input and errors for inference.modelapi in case of error ([#193](https://github.com/monocle2ai/monocle/pull/193))
- Removed special handling for streaming in wrapper ([#192](https://github.com/monocle2ai/monocle/pull/192))


- Add Span error handling ([#186](https://github.com/monocle2ai/monocle/pull/186))
- Add teams ai enhancements ([#184](https://github.com/monocle2ai/monocle/pull/184))


- Added conversation id in scope for teams ai bot ([#180](https://github.com/monocle2ai/monocle/pull/180))
- Update inference entity type of TeamsAI SDK ([#178](https://github.com/monocle2ai/monocle/pull/178))
- Added stream and async for openai ([#177](https://github.com/monocle2ai/monocle/pull/177))
- Update inference span of TeamsAI ([#176](https://github.com/monocle2ai/monocle/pull/176))
- Remove Preset span name and Bugfix for Event ([#175](https://github.com/monocle2ai/monocle/pull/175))
- Add haystack anthropic sample ([#174](https://github.com/monocle2ai/monocle/pull/174))
- aiohttp auto instrumentation ([#173](https://github.com/monocle2ai/monocle/pull/173))
- Add source path to spans and fix json syntax in file exporter ([#172](https://github.com/monocle2ai/monocle/pull/172))
- Added changes for openai streaming ([#171](https://github.com/monocle2ai/monocle/pull/171))
- Add llama index anthropic sample ([#170](https://github.com/monocle2ai/monocle/pull/170))

## Version 0.3.1 (2024-04-18)

- Add MetaModel for Anthropic SDK ([#159](https://github.com/monocle2ai/monocle/pull/159))
- Add openAI response for openAI and AzureOpenAI ([#158](https://github.com/monocle2ai/monocle/pull/158))
- Update retrieval span for Boto Client ([#157](https://github.com/monocle2ai/monocle/pull/157))
- Resolve token threshold error ([#156](https://github.com/monocle2ai/monocle/pull/156))
- Update Inference Span ([#155](https://github.com/monocle2ai/monocle/pull/155))
- Refactor workflow and spans ([#160](https://github.com/monocle2ai/monocle/pull/160))
- Support monocle exporter list as parameter to `setup_monocle_telemetry()` ([#161](https://github.com/monocle2ai/monocle/pull/161))
- Add langchain anthropic sample ([#165](https://github.com/monocle2ai/monocle/pull/165))

## Version 0.3.0 (2024-03-18)

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

- Add support for teams finish type, move finish types to metadata section ([#240](https://github.com/monocle2ai/monocle/pull/240))
- Fix integration tests ([#239](https://github.com/monocle2ai/monocle/pull/239))
- Add function_name attribute for Azure function ([#238](https://github.com/monocle2ai/monocle/pull/238))
- Added MCP and A2A ([#237](https://github.com/monocle2ai/monocle/pull/237))
- Add Metamodel for Lambda Func ([#236](https://github.com/monocle2ai/monocle/pull/236))
- Allow llm SDKs as workflow types ([#234](https://github.com/monocle2ai/monocle/pull/234))
- Add Sample Span for VertexAI ([#233](https://github.com/monocle2ai/monocle/pull/233))
- Add Haystack Gemini sample ([#232](https://github.com/monocle2ai/monocle/pull/232))
- Add Llama Index Gemini sample ([#231](https://github.com/monocle2ai/monocle/pull/231))
- Add workflow span name to all spans ([#230](https://github.com/monocle2ai/monocle/pull/230))
- Add Langchain Gemini sample ([#229](https://github.com/monocle2ai/monocle/pull/229))
- Add Embeddings Span in Gemini Metamodel ([#228](https://github.com/monocle2ai/monocle/pull/228))
- Agent metamodel updates for LangGraph ([#227](https://github.com/monocle2ai/monocle/pull/227))
- Add fastapi metamodel with fastapi tracid propogation testcase ([#226](https://github.com/monocle2ai/monocle/pull/226))


- Fix missing inference span in google ADK ([#249](https://github.com/monocle2ai/monocle/pull/249))
- Added openai agents instrumentation ([#248](https://github.com/monocle2ai/monocle/pull/248))
- Google Agent development kit meta mode ([#247](https://github.com/monocle2ai/monocle/pull/247))
- Add Support for LiteLLM OpenAI and AzureOpenAI ([#246](https://github.com/monocle2ai/monocle/pull/246))
- Added inference subtype for langgrap, openai and anthropic ([#245](https://github.com/monocle2ai/monocle/pull/245))
- Add support for teams finish type, move finish types to metadata section ([#240](https://github.com/monocle2ai/monocle/pull/240))
- Fix integration tests ([#239](https://github.com/monocle2ai/monocle/pull/239))
- Add function_name attribute for Azure function ([#238](https://github.com/monocle2ai/monocle/pull/238))
- Added MCP and A2A ([#237](https://github.com/monocle2ai/monocle/pull/237))
- Add Metamodel for Lambda Func ([#236](https://github.com/monocle2ai/monocle/pull/236))
- Allow llm SDKs as workflow types ([#234](https://github.com/monocle2ai/monocle/pull/234))
- Add Sample Span for VertexAI ([#233](https://github.com/monocle2ai/monocle/pull/233))
- Add Haystack Gemini sample ([#232](https://github.com/monocle2ai/monocle/pull/232))
- Add Llama Index Gemini sample ([#231](https://github.com/monocle2ai/monocle/pull/231))
- Add workflow span name to all spans ([#230](https://github.com/monocle2ai/monocle/pull/230))
- Add Langchain Gemini sample ([#229](https://github.com/monocle2ai/monocle/pull/229))
- Add Embeddings Span in Gemini Metamodel ([#228](https://github.com/monocle2ai/monocle/pull/228))
- Agent metamodel updates for LangGraph ([#227](https://github.com/monocle2ai/monocle/pull/227))
- Add fastapi metamodel with fastapi tracid propogation testcase ([#226](https://github.com/monocle2ai/monocle/pull/226))


- Monocle testing framework ([#288](https://github.com/monocle2ai/monocle/pull/288))
- Fix package version conflict ([#293](https://github.com/monocle2ai/monocle/pull/293))
- Remove hugging face direct dependency on Monocle. Also update the tests
- Hugging face inference test ([#290](https://github.com/monocle2ai/monocle/pull/290))
- Attributes fixes ([#289](https://github.com/monocle2ai/monocle/pull/289))
- mistral embed instrumentation ([#279](https://github.com/monocle2ai/monocle/pull/279))

