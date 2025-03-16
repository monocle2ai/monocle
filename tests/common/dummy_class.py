from opentelemetry.trace import Tracer
from monocle_apptrace.instrumentation.common.utils import with_tracer_wrapper
from monocle_apptrace.instrumentation.common.instrumentor import monocle_trace_scope_method
SCOPE_NAME="test_scope1"
SCOPE_VALUE="test1"

@with_tracer_wrapper
def dummy_wrapper(tracer: Tracer, handler, to_wrap, wrapped, instance, args, kwargs):
    if callable(to_wrap.get("span_name_getter")):
        name = to_wrap.get("span_name_getter")(instance)
    elif hasattr(instance, "name") and instance.name:
        name = f"{to_wrap.get('span_name')}.{instance.name.lower()}"
    elif to_wrap.get("span_name"):
        name = to_wrap.get("span_name")
    else:
        name = f"dummy.{instance.__class__.__name__}"
    kind = to_wrap.get("kind")
    with tracer.start_as_current_span(name) as span:
        return_value = wrapped(*args, **kwargs)

    return return_value

class DummyClass:
    def dummy_method(val: int):
        print("entering dummy_method: " + str(val))

    def dummy_chat(self, prompt: str):
        pass

    def dummy_error(self, prompt:str):
        raise Exception("dummy error for "+ prompt)

    async def add3(self, val:int, raise_error:bool=False):
        if raise_error:
            raise Exception(f"Dummy aysnc error {val}")
        return val + 3

    async def add2(self, val:int, raise_error:bool = False):
        return await self.add3(val, raise_error) + 2

    async def add1(self, val:int, raise_error:bool=False):
        return await self.add2(val, raise_error) + 1

    async def dummy_async_error(self, prompt:str):
        raise Exception("dummy async error for "+ prompt)

    @monocle_trace_scope_method(SCOPE_NAME, SCOPE_VALUE)
    async def scope_async_decorator_test_method(self):
        return await self.add1(10)

    async def scope_async_config_test_method(self):
        return await self.add1(10)
    
