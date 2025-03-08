# pylint: disable=too-few-public-methods
from typing import Any, Dict
from monocle_apptrace.instrumentation.common.wrapper import task_wrapper, scope_wrapper
from monocle_apptrace.instrumentation.common.span_handler import SpanHandler, NonFrameworkSpanHandler
from monocle_apptrace.instrumentation.metamodel.botocore.methods import BOTOCORE_METHODS
from monocle_apptrace.instrumentation.metamodel.botocore.handlers.botocore_span_handler import BotoCoreSpanHandler
from monocle_apptrace.instrumentation.metamodel.langchain.methods import (
    LANGCHAIN_METHODS,
)
from monocle_apptrace.instrumentation.metamodel.llamaindex.methods import (LLAMAINDEX_METHODS, )
from monocle_apptrace.instrumentation.metamodel.haystack.methods import (HAYSTACK_METHODS, )
from monocle_apptrace.instrumentation.metamodel.openai.methods import (OPENAI_METHODS,)
from monocle_apptrace.instrumentation.metamodel.langgraph.methods import LANGGRAPH_METHODS
from monocle_apptrace.instrumentation.metamodel.flask.methods import (FLASK_METHODS, )
from monocle_apptrace.instrumentation.metamodel.flask._helper import FlaskSpanHandler
from monocle_apptrace.instrumentation.metamodel.requests.methods import (REQUESTS_METHODS, )
from monocle_apptrace.instrumentation.metamodel.requests._helper import RequestSpanHandler

class WrapperMethod:
    def __init__(
            self,
            package: str,
            object_name: str,
            method: str,
            span_name: str = None,
            output_processor : str = None,
            wrapper_method = task_wrapper,
            span_handler = 'default',
            scope_name: str = None,
            span_type: str = None
            ):
        self.package = package
        self.object = object_name
        self.method = method
        self.span_name = span_name
        self.output_processor=output_processor
        self.span_type = span_type

        self.span_handler:SpanHandler.__class__ = span_handler
        self.scope_name = scope_name
        if scope_name:
            self.wrapper_method = scope_wrapper
        else:
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
            'span_handler': self.span_handler,
            'scope_name': self.scope_name,
            'span_type': self.span_type
        }
        return instance_dict

    def get_span_handler(self) -> SpanHandler:
        return self.span_handler()

DEFAULT_METHODS_LIST = LANGCHAIN_METHODS + LLAMAINDEX_METHODS + HAYSTACK_METHODS + BOTOCORE_METHODS + FLASK_METHODS + REQUESTS_METHODS + LANGGRAPH_METHODS + OPENAI_METHODS

MONOCLE_SPAN_HANDLERS: Dict[str, SpanHandler] = {
    "default": SpanHandler(),
    "botocore_handler": BotoCoreSpanHandler(),
    "flask_handler": FlaskSpanHandler(),
    "request_handler": RequestSpanHandler(),
    "non_framework_handler": NonFrameworkSpanHandler()
}
