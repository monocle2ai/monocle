from opentelemetry.trace import Tracer
from monocle_apptrace.utils import with_tracer_wrapper

@with_tracer_wrapper
def dummy_wrapper(tracer: Tracer, to_wrap, wrapped, instance, args, kwargs):
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
