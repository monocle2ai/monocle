# monocle genAI observability
### Background
Generative AI (GenAI) is the type of AI used to create content such as conversations, images, or video based on prior learning from existing content. GenAI relies on foundational models, which are exceptionally large ML models trained on vast amounts of generalized and unlabeled data to perform variety of general tasks such as understanding language and generating new text, audio or images from user provided prompts in a human language. Foundational models (FM) work by using learned patterns and relationships from the training data to predict the next item in a sequence given a prompt. It is cheaper and faster for data scientists to use foundational models as starting points rather than building models from scratch to build ML apps.  
Large Language Models (LLMs) are a class of foundational models trained on text data used to perform a variety of tasks such as understanding language, reasoning over text, and generating new text based on user prompts in a human language. Examples of LLMs include ChatGPT, Llama, and Claude. 
LLM-based AI apps leverage understanding language, reasoning & text generation to augment or automate complex tasks that typically require human intervention such as summarizing legal documents, triaging customer support tickets, or more.  
Typically, AI developers build LLM-based AI apps that automate complex workflows by combining multiple LLMs and components such as prompts, vectors, or agents that each solve a discrete task that are connected by chains or pipelines in different ways using LLM (Large Language Model) orchestration frameworks.  
When deployed to production, different parts of multi-component distributed LLM-based AI apps run on a combination of different kinds of AI infrastructure such as LLM-as-a-Service, GPU (graphics processing units) clouds, managed services from cloud, or custom-engineered AI stack. Typically, these systems are managed in production by IT DevOps engineers.  
AI developers code, monitor, debug and optimize the resources in an LLM-based AI application. IT DevOps engineers monitor, troubleshoot, and optimize the services in the AI infra that the LLM-based AI application runs on. 

## Introducing “Monocle – An eye for A.I.”
The goal of project Monocle is to help GenAI developer to trace their applications. A typical GenAI application comprises of several technology components like application code/workflow, models, inferences services, vector databases etc. Understanding the dependencies and tracking application quickly becomes a difficult task. Monocle can be integrated into application code with very little to no code changes. Monocle supports tracing all GenAI technology components, application frameworks, LLM hosting services. We do all the hard work of finding what needs to be instrumented and how to instrument it. This enables the enlightened applications to generate detailed traces without any additional efforts from the developers.
The traces are compatible with OpenTelemetry format. They are further enriched to contain lot more attribute relevant to GenAI applications like prompts. The project will have out of box support to store the traces locally and a extensibility for a third party store which can be implemented by end user or a supplied by third party vendors.    

## Monocle integration
### genAI Appliation frameworks
- Langchain
- LlamaIndex
- Haystack
### LLMs
- OpenAI
- Azure OpenAI
- NVIDIA Triton

## Getting started
### Try Monocle with your python genAI application
- Get latest Monocle python brary 
```
    pip install monocle_apptrace
```
- Enable Monocle tracing in your app by adding following
```
    setup_okahu_telemetry(workflow_name="your-app-name")
```
Please refer to [Monocle user guide](Monocle_User_Guide.md) for more details

## Get involved
### Provide feedback
- Submit issues and enhancements requests via Github issues

### Contribute
- Monocle is community based open source project. We welcome your contributions. Please refer to the CONTRIBUTING and CODE_OF_CONDUCT for guidelines. The [contributor's guide](CONTRIBUTING.md) provides technical details of the project.

