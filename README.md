# monocle genAI observability

This package provides monocle telemetry setup.

## Installing the core and dev dependencies
```
> pip install .
> pip install -e ".[dev]"

> python3 -m pip install pipenv
> pipenv install build

```

## Building the package

```
> python3 -m build 
```
## Publishing the package

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

