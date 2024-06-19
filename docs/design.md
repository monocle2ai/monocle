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










