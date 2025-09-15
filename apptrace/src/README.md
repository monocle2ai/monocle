## Monocle Concepts

### Traces
Traces are the full view of a single end-to-end application execution. 

Examples of traces include one response to end user’s question by a chatbot app. Traces consists of various metadata about the application run including status, start time, duration, input/outputs etc. It also includes a list of individual steps aka “spans with details about that step.It’s typically the workflow code components of an application that generate the traces for application runs. 

Traces are collections of spans. 

### Spans
Spans are the individual steps executed by the application to perform a GenAI related task.

Examples of spans include app retrieving vectors from DB, app querying LLM for inference etc. The span includes the type of operation, start time, duration and metadata relevant to that step eg Model name, parameters and model endpoint/server for an inference request.

## Contribute to Monocle 

Monocle includes:
- Methods for instrumentation of app code 
  - Base code for wrapping methods of interest in included in current folder
  - Framework specific code is organized in a folder with the framework name
- Metamodel for how attributes and events for GenAI components are represented in OpenTelemety format
  - See [metamodel](./metamodel/README.md) for supported GenAI entities, how functions operating on those entities map to spans and format of spans 
- Exporters to send trace data to various locations. See [exporters](./exporters)

See [Monocle committer guide](/Monocle_committer_guide.md). 

## Get Monocle

Option 1 - Download released packages from Pypi
``` 
    python3 -m pip install pipenv
    pip install monocle-apptrace
```

Option 2 - Build and install locally from source
```
    pip install .
    pip install -e ".[dev]"

    python3 -m pip install pipenv
    pipenv install build
```

## Examples of app instrumentation with Monocle
 
### apps written using LLM orchestration frameworks 

```python
from langchain.chains import LLMChain
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate

# Import the monocle_apptrace instrumentation method 
from monocle_apptrace.instrumentor import setup_monocle_telemetry
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

# Call the setup Monocle telemetry method
setup_monocle_telemetry(workflow_name = "simple_math_app",
        span_processors=[BatchSpanProcessor(ConsoleSpanExporter())])

llm = OpenAI()
prompt = PromptTemplate.from_template("1 + {number} = ")

chain = LLMChain(llm=llm, prompt=prompt)
chain.invoke({"number":2})

# Trace is generated when invoke() method is called

```

### apps with custom methods

```python

# Import the monocle_apptrace instrumentation method
from monocle_apptrace.wrapper import WrapperMethod,task_wrapper,atask_wrapper
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

# Extend the default wrapped methods list as follows
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

# Trace is generated when the invoke() method is called in langchain.schema.runnable package

```
