## Version 0.3.0b1 (2024-12-10)

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
