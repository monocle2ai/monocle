## How member functions are referenced in python classes

A sample python class looks as following

```python
class MyClass
    def my_function():
        # do something here
```

All the member functions of classes are stored in a attribute called "\_\_dict__".

So when we write in python 
```python
MyClass().my_function()
```

Under the hood, python finds my_function in MyClass.\_\_dict__ and calls it.

## How we intercept the my_function class function for instrumentation

To intercept my_function, our SDK replaces my_function in MyClass.\_\_dict__ with our special wrapper.

This wrapper helps us in instrumentation by : 
- measuring time it took for the executing this function.
- details of the arguments with which the function was called
- we also get access to the class instance and can extract any of the class fields.
- the value returned by the function

## Example

Let us understand by using a LLM class.
So for the following example

```python
class My_LLM():
    model_name = "gpt-4"

    def call_llm(prompt_text):
        llm_response = make_api_call(prompt_text)

        return llm_response
```
We replace **My_LLM.\_\_dict__.call_llm** with our instrumentation wrapper.

The wrapper then performs the following tasks:
- extracts the llm **model_name** from the class instance.

- extracts the llm prompt from the **prompt_text** argument of call_llm.

- extracts the llm response from the **llm_response** value returned.


## How to user add custom attributes in monocle's
Monocle provides users with the ability to add custom attributes to various spans, such as inference and retrieval spans, by utilizing the output processor within its metamodel. This feature allows for dynamic attribute assignment through lambda functions, which operate on an arguments dictionary.
The arguments dictionary contains key-value pairs that can be used to compute custom attributes. The dictionary includes the following components: 
```python
arguments = {"instance":instance, "args":args, "kwargs":kwargs, "output":return_value}
```
By leveraging this dictionary, users can define custom attributes for spans, enabling the integration of additional context and information into the tracing process. The lambda functions used in the attributes field can access and process these values to enrich the span with relevant custom data.

### Example JSON for custom attributes
```json
{
  "type": "retrieval",
  "attributes": [
    [
      {
        "_comment": "vector store name and type",
        "attribute": "name",
        "accessor": "lambda arguments: type(arguments['instance'].vectorstore).__name__"
      },
      {
        "attribute": "type",
        "accessor": "lambda arguments: 'vectorstore.'+type(arguments['instance'].vectorstore).__name__"
      }
    ]
  ]
}

```
### Key Components of the JSON Format:
- type: Specifies the type of operation or task. This is optional but can be used to categorize the span.
- attributes: An array of arrays, where each inner array contains definitions for multiple attributes. Each attribute should have:
  -  _comment: (Optional) A description of the attribute.
  - attribute: The name of the span attribute to set.
  - accessor: A lambda function or expression that evaluates the value for the attribute. The accessor can access the function's arguments like instance, args, kwargs, and return_value.

### Adding Custom Attributes
To add custom attributes:

1. Define a new entry in the attributes list for the task you're working on.
2. Provide an attribute name (this will be the span attribute name).
3. Create an accessor (using a lambda function or expression) that pulls the relevant data from the arguments.
4. Ensure the custom attributes are included in the output_processor section of the to_wrap dictionary when calling the process_span function.








