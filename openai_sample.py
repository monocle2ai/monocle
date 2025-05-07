from openai import OpenAI, AsyncOpenAI
import os
import sys

from monocle_apptrace import start_trace, stop_trace
from monocle_apptrace.instrumentation.common.wrapper import atask_wrapper, task_wrapper
os.environ["OPENAI_API_KEY"] = ""
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from monocle_apptrace import setup_monocle_telemetry

setup_monocle_telemetry(
    workflow_name="openai_sample",
    wrapper_methods=[
    ],
)

async def async_main(): 
    # Check if the OpenAI API key is
    # set in the environment variables
    if "OPENAI_API_KEY" not in os.environ:
        print("Please set the OPENAI_API_KEY environment variable.")
        return

    # Initialize the OpenAI API client
    openai = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    # creating the stream
    # responses and completions API
    
    token = start_trace()
    stream = await openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello, how are you? Answer what is coffee in 10 words."}],
        stream=True,
        stream_options={
           "include_usage": True,
        },
    )

    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta:
            print(chunk.choices[0].delta.content, end="", flush=True)
    
    stop_trace(token)


def main():
    # Check if the OpenAI API key is
    # set in the environment variables
    if "OPENAI_API_KEY" not in os.environ:
        print("Please set the OPENAI_API_KEY environment variable.")
        return

    # Initialize the OpenAI API client
    openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    # creating the stream
    # responses and completions API
    
    token = start_trace()
    stream = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello, how are you? Answer what is coffee in 1000 words."}],
        stream=True,
        stream_options={
           "include_usage": True,
        },
    )
    # have buffer for accumulating the stream, use configuration for that
    
    # assuming that we can capture all events from stream
    
    # data.input . response.created from stream
    # stream: tiemstamp of the event , name of the openai event 
    # data.output: output completed time , whatever has been received from the stream will be put inoutput
    
    # metadata: response.completed in metadata event tiemstamp
    # stream close : this is when the span has ended and flushed
    
    # if the stream is closed,close the span and mark the span as OK
    # or if the response completed event is received, close the span as unset
    # or if the error event is received, close the span and mark it as error
    
    # stream = true works for teams
    # check if stream uses openai, and the trace is captured

    # stream = false should capture the trace
    
    
    
    
    
    
    # stream.close()
    # stream state .....
    
    # openai => event1 => 1 seconds=> event2 1 second=> event3

    # consumtion of stream the response from the API
    # event if a stream receives a completed event, event if stream is not consumed, 
    # we should have all events
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta:
            print(chunk.choices[0].delta.content, end="", flush=True)
    
    stop_trace(token)
    
    # captring metadata at the end
    # making sure that the span is not exported until the stream has ended


# main()
if __name__ == "__main__":
    # main()
    import asyncio
    asyncio.run(async_main())
