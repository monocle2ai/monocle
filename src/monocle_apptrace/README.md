#Monocle User Guide

## Monocle Concepts
### Traces
Traces are the full view of a single end-to-end application KPI eg Chatbot application to provide a response to end user’s question. Traces consists of various metadata about the application run including status, start time, duration, input/outputs etc. It also includes a list of individual steps aka “spans with details about that step.
It’s typically the workflow code components of an application that generate the traces for application runs. 
### Spans
Spans are the individual steps executed by the application to perform a GenAI related task” eg app retrieving vectors from DB, app querying LLM for inference etc. The span includes the type of operation, start time, duration and metadata relevant to that step eg Model name, parameters and model endpoint/server for an inference request.
It’s typically the workflow code components of an application that generate the traces for application runs.

## Setup Monocle
- You can download Monocle library releases from Pypi
``` 
    > python3 -m pip install pipenv
    > pip install monocle-observability
```
- You can locally build and install Monocle library from source
```
> pip install .
> pip install -e ".[dev]"

> python3 -m pip install pipenv
> pipenv install build
```

## Examples 
### Enable Monocle tracing in your application
```python
from monocle_apptrace.instrumentor import setup_monocle_telemetry
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from langchain.chains import LLMChain
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate

# Call the setup Monocle telemetry method
setup_monocle_telemetry(workflow_name = "simple_math_app",
        span_processors=[BatchSpanProcessor(ConsoleSpanExporter())])

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
                object_name="RunnableParallel",
                method="invoke",
                span_name="langchain.workflow",
                wrapper=task_wrapper),
            WrapperMethod(
                package="langchain.schema.runnable",
                object_name="RunnableParallel",
                method="ainvoke",
                span_name="langchain.workflow",
                wrapper=atask_wrapper)
        ])

```