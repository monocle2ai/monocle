# pylint: disable=too-few-public-methods
from monocle_apptrace.instrumentation.common.wrapper import task_wrapper
from monocle_apptrace.instrumentation.metamodel.botocore.methods import BOTOCORE_METHODS
from monocle_apptrace.instrumentation.metamodel.langchain.methods import (
    LANGCHAIN_METHODS,
)
from monocle_apptrace.instrumentation.metamodel.llamaindex.methods import (LLAMAINDEX_METHODS, )
from monocle_apptrace.instrumentation.metamodel.haystack.methods import (HAYSTACK_METHODS, )


class WrapperMethod:
    def __init__(
            self,
            package: str,
            object_name: str,
            method: str,
            span_name: str = None,
            output_processor : str = None,
            wrapper_method = task_wrapper,
            span_handler = 'default'
            ):
        self.package = package
        self.object = object_name
        self.method = method
        self.span_name = span_name
        self.output_processor=output_processor
        self.span_handler = span_handler

        self.wrapper_method = wrapper_method

    def to_dict(self) -> dict:
        # Create a dictionary representation of the instance
        instance_dict = {
            'package': self.package,
            'object': self.object,
            'method': self.method,
            'span_name': self.span_name,
            'output_processor': self.output_processor,
            'wrapper_method': self.wrapper_method,
            'span_handler': self.span_handler
        }
        return instance_dict


DEFAULT_METHODS_LIST = LANGCHAIN_METHODS + LLAMAINDEX_METHODS + HAYSTACK_METHODS + BOTOCORE_METHODS
