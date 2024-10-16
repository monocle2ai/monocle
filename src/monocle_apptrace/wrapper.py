
from monocle_apptrace.haystack import HAYSTACK_METHODS
from monocle_apptrace.langchain import LANGCHAIN_METHODS
from monocle_apptrace.llamaindex import LLAMAINDEX_METHODS
from monocle_apptrace.wrap_common import task_wrapper

# pylint: disable=too-few-public-methods
class WrapperMethod:
    def __init__(
            self,
            package: str,
            object_name: str,
            method: str,
            span_name: str = None,
            output_processor : list[str] = None,
            wrapper = task_wrapper
            ):
        self.package = package
        self.object = object_name
        self.method = method
        self.span_name = span_name
        self.output_processor=output_processor

        self.wrapper = wrapper

INBUILT_METHODS_LIST = LANGCHAIN_METHODS + LLAMAINDEX_METHODS + HAYSTACK_METHODS
