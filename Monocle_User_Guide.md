#Monocle User Guide

## Monocle Concepts
### Traces
Traces are the full view of a single end-to-end application KPI, for example Chatbot application to provide a response to end user’s question. Traces consist of various metadata about the application run including status, start time, duration, input/outputs etc. They also include a list of individual steps aka “spans with details about that step.
It’s typically the workflow code components of an application that generate the traces for application runs. 
### Spans
Spans are the individual steps executed by the application to perform a GenAI related task”, for example app retrieving vectors from DB, app querying LLM for inference etc. The span includes the type of operation, start time, duration and metadata relevant to that step e.g., Model name, parameters and model endpoint/server for an inference request.
It’s typically the workflow code components of an application that generate the traces for application runs.

## Setup Monocle
- You can download Monocle library releases from Pypi
``` 
    > pip install monocle_apptrace
```
- You can locally build and install Monocle library from source
```
> pip install .
```
- Install the optional test dependencies listed against dev in pyproject.toml in editable mode
```
> pip install -e ".[dev]"
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