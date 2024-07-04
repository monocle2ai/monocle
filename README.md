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

## Monocle Concepts
### Application
An application is a business concept. It is a set of different logical components combined by some code/workflow to deliver some business KPI. Each of these logical components will be hosted on an infrastructure component. The application will have a unique name/id, friendly name and description. The application could include additional business metadata.
### Components
Components are the core building blocks of the user’s AI ecosystem. Every component has a unique identifier, a friendly name, description and type. A component can optionally include a set of attributes relevant to that component type eg.  Triton server will specify the connection endpoint or Model will include parameters etc.
#### frastructure component
An infrastructure component is an instance of a service eg NVIDIA Triton server instance or Kubernetes cluster instance or Postgres database server etc. Every instance of a service type is a separate component eg RDS Postgres Finance-Service-EastUS and Sales-Service-WestUS are both Postgres servers and two different components. 
#### Logical component
A logical component is a piece of code or data eg workflow code, GPT 3.5 Turbo model, vector data set etc. Every copy of such code or data is separate component.
Dependencies between components
A notion of dependency exists between components. A component can be ‘hosted’ by another component eg a Model is hosted on inference server; the inference server is hosted on Kubernetes. Note that a logical component can be hosted on an infrastructure component but can’t host any other component.
Linking logical components within an application
Each application will have a set of logical components stitched together. The output of one logical component is consumed by another, eg Langchain workflow will use a foundational model to consume the inference. Note that a given logical component could be used in more than one application eg a GPT 3 model is used by two different applications.
### Traces
Traces are the full view of a single end-to-end application KPI eg Chatbot application to provide a response to end user’s question. Traces consists of various metadata about the application run including status, start time, duration, input/outputs etc. It also includes a list of individual steps aka “spans with details about that step.
It’s typically the workflow code components of an application that generate the traces for application runs. 
### Spans
Spans are the individual steps executed by the application to perform a GenAI related task” eg app retrieving vectors from DB, app querying LLM for inference etc. The span includes the type of operation, start time, duration and metadata relevant to that step eg Model name, parameters and model endpoint/server for an inference request.
It’s typically the workflow code components of an application that generate the traces for application runs.

## Setup
### Installing the core and dev dependencies
```
> pip install .
> pip install -e ".[dev]"

> python3 -m pip install pipenv
> pipenv install build

```

### Building the package

```
> python3 -m build 
```
### Publishing the package

```
> python3 -m pip install --upgrade twine
> python3 -m twine upload --repository testpypi dist/*
```

## Installing the package

The steps to set the credential can be found here:
https://packaging.python.org/en/latest/specifications/pypirc/

After setup of credentials, follow the commands below to publish the package to testpypi:

```
> python3 -m pip install pipenv
> pipenv install monocle-observability
```

## References

[Managing application dependencies](https://packaging.python.org/en/latest/tutorials/managing-dependencies/)

## Usage
```python
from monocle_apptrace.instrumentor import setup_monocle_telemetry
from langchain.chains import LLMChain
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate

# Call the setup Monocle telemetry method
app_name = "simple_math_app"
setup_monocle_telemetry(workflow_name=app_name)

llm = OpenAI()
prompt = PromptTemplate.from_template("1 + {number} = ")

chain = LLMChain(llm=llm, prompt=prompt)
chain.invoke({"number":2})

# Request callbacks: Finally, let's use the request `callbacks` to achieve the same result
chain = LLMChain(llm=llm, prompt=prompt)
chain.invoke({"number":2}, {"callbacks":[handler]})
    
```

### Monitoring custom methods with Monocle

```python
from monocle_apptrace.wrapper import WrapperMethod,task_wrapper,atask_wrapper
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

# extend the default wrapped methods list as follows
app_name = "simple_math_app"
setup_monocle_telemetry(
        workflow_name=app_name,
        span_processors=[BatchSpanProcessor(ConsoleSpanExporter())],
        wrapper_methods=[
            WrapperMethod(
                package="langchain.schema.runnable",
                object="RunnableParallel",
                method="invoke",
                span_name="langchain.workflow",
                wrapper=task_wrapper),
            WrapperMethod(
                package="langchain.schema.runnable",
                object="RunnableParallel",
                method="ainvoke",
                span_name="langchain.workflow",
                wrapper=atask_wrapper)
        ])

```

