import logging
import os
from importlib.metadata import version
from opentelemetry.context import get_value, set_value, attach, detach
from opentelemetry.sdk.trace import Span
from opentelemetry.trace.status import Status, StatusCode
from monocle_apptrace.instrumentation.common.constants import (
    QUERY,
    service_name_map,
    service_type_map,
    MONOCLE_SDK_VERSION
)
from monocle_apptrace.instrumentation.common.utils import set_attribute, get_scopes, MonocleSpanException
from monocle_apptrace.instrumentation.common.constants import WORKFLOW_TYPE_KEY, WORKFLOW_TYPE_GENERIC

logger = logging.getLogger(__name__)

WORKFLOW_TYPE_MAP = {
    "llama_index": "workflow.llamaindex",
    "langchain": "workflow.langchain",
    "haystack": "workflow.haystack"
}

class SpanHandler:

    def __init__(self,instrumentor=None):
        self.instrumentor=instrumentor

    def set_instrumentor(self, instrumentor):
        self.instrumentor = instrumentor

    def validate(self, to_wrap, wrapped, instance, args, kwargs):
        pass

    def pre_tracing(self, to_wrap, wrapped, instance, args, kwargs):
        pass

    def post_tracing(self, to_wrap, wrapped, instance, args, kwargs, return_value):
        pass

    def skip_span(self, to_wrap, wrapped, instance, args, kwargs) -> bool:
        # If this is a workflow span type and a workflow span is already generated, then skip generating this span
        if to_wrap.get('span_type') == "workflow" and self.is_workflow_span_active():
            return True
        return False

    def pre_task_processing(self, to_wrap, wrapped, instance, args,kwargs, span):
        if "pipeline" in to_wrap['package']:
            set_attribute(QUERY, args[0]['prompt_builder']['question'])

    @staticmethod
    def set_default_monocle_attributes(span: Span):
        """ Set default monocle attributes for all spans """
        try:
            sdk_version = version("monocle_apptrace")
            span.set_attribute(MONOCLE_SDK_VERSION, sdk_version)
        except Exception as e:
            logger.warning("Exception finding monocle-apptrace version.")
        for scope_key, scope_value in get_scopes().items():
            span.set_attribute(f"scope.{scope_key}", scope_value)

    @staticmethod
    def set_workflow_properties(span: Span, to_wrap = None):
        """ Set attributes of workflow if this is a root span"""
        SpanHandler.set_workflow_attributes(to_wrap, span)
        SpanHandler.set_app_hosting_identifier_attribute(span)
        span.set_status(StatusCode.OK)


    def post_task_processing(self, to_wrap, wrapped, instance, args, kwargs, result, span:Span):
        if span.status.status_code == StatusCode.UNSET:
            span.set_status(StatusCode.OK)

    def hydrate_span(self, to_wrap, wrapped, instance, args, kwargs, result, span):
        self.hydrate_attributes(to_wrap, wrapped, instance, args, kwargs, result, span)
        self.hydrate_events(to_wrap, wrapped, instance, args, kwargs, result, span)

    def hydrate_attributes(self, to_wrap, wrapped, instance, args, kwargs, result, span):
        span_index = 0
        if SpanHandler.is_root_span(span):
            span_index = 2 # root span will have workflow and hosting entities pre-populated
        if 'output_processor' in to_wrap and to_wrap["output_processor"] is not None:    
            output_processor=to_wrap['output_processor']
            if 'type' in output_processor:
                        span.set_attribute("span.type", output_processor['type'])
            else:
                logger.warning("type of span not found or incorrect written in entity json")
            if 'attributes' in output_processor:
                for processors in output_processor["attributes"]:
                    for processor in processors:
                        attribute = processor.get('attribute')
                        accessor = processor.get('accessor')

                        if attribute and accessor:
                            attribute_name = f"entity.{span_index+1}.{attribute}"
                            try:
                                arguments = {"instance":instance, "args":args, "kwargs":kwargs, "result":result}
                                result = accessor(arguments)
                                if result and isinstance(result, (str, list)):
                                    span.set_attribute(attribute_name, result)
                            except MonocleSpanException as e:
                                span.set_status(StatusCode.ERROR, e.message)
                            except Exception as e:
                                logger.debug(f"Error processing accessor: {e}")
                        else:
                            logger.debug(f"{' and '.join([key for key in ['attribute', 'accessor'] if not processor.get(key)])} not found or incorrect in entity JSON")
                    span_index += 1
            else:
                logger.debug("attributes not found or incorrect written in entity json")

        # set scopes as attributes by calling get_scopes()
        # scopes is a Mapping[str:object], iterate directly with .items()
        for scope_key, scope_value in get_scopes().items():
            span.set_attribute(f"scope.{scope_key}", scope_value)
        
        if span_index > 0:
            span.set_attribute("entity.count", span_index)


    def hydrate_events(self, to_wrap, wrapped, instance, args, kwargs, result, span):
        if 'output_processor' in to_wrap and to_wrap["output_processor"] is not None:
            output_processor=to_wrap['output_processor']
            arguments = {"instance": instance, "args": args, "kwargs": kwargs, "result": result}
            if 'events' in output_processor:
                events = output_processor['events']
                for event in events:
                    event_name = event.get("name")
                    event_attributes = {}
                    attributes = event.get("attributes", [])
                    for attribute in attributes:
                        attribute_key = attribute.get("attribute")
                        accessor = attribute.get("accessor")
                        if accessor:
                            try:
                                if attribute_key is not None:
                                    event_attributes[attribute_key] = accessor(arguments)
                                else:
                                    event_attributes.update(accessor(arguments))
                            except MonocleSpanException as e:
                                span.set_status(StatusCode.ERROR, e.message)
                            except Exception as e:
                                logger.debug(f"Error evaluating accessor for attribute '{attribute_key}': {e}")
                    span.add_event(name=event_name, attributes=event_attributes)

    @staticmethod
    def set_workflow_attributes(to_wrap, span: Span):
        span_index = 1
        workflow_name = SpanHandler.get_workflow_name(span=span)
        if workflow_name:
            span.set_attribute("span.type", "workflow")
            span.set_attribute(f"entity.{span_index}.name", workflow_name)
        workflow_type = SpanHandler.get_workflow_type(to_wrap)
        span.set_attribute(f"entity.{span_index}.type", workflow_type)

    @staticmethod
    def get_workflow_type(to_wrap):
        # workflow type
        workflow_type = WORKFLOW_TYPE_GENERIC
        if to_wrap is not None:
            package_name = to_wrap.get('package')
            for (package, framework_workflow_type) in WORKFLOW_TYPE_MAP.items():
                if (package_name is not None and package in package_name):
                    workflow_type = framework_workflow_type
                    break
        return workflow_type

    def set_app_hosting_identifier_attribute(span):
        span_index = 2
        # Search env to indentify the infra service type, if found check env for service name if possible
        span.set_attribute(f"entity.{span_index}.type", f"app_hosting.generic")
        span.set_attribute(f"entity.{span_index}.name", "generic")
        for type_env, type_name in service_type_map.items():
            if type_env in os.environ:
                span.set_attribute(f"entity.{span_index}.type", f"app_hosting.{type_name}")
                entity_name_env = service_name_map.get(type_name, "unknown")
                span.set_attribute(f"entity.{span_index}.name", os.environ.get(entity_name_env, "generic"))

    @staticmethod
    def get_workflow_name(span: Span) -> str:
        try:
            return get_value("workflow_name") or span.resource.attributes.get("service.name")
        except Exception as e:
            logger.exception(f"Error getting workflow name: {e}")
            return None

    @staticmethod
    def is_root_span(curr_span: Span) -> bool:
        try:
            if curr_span is not None and hasattr(curr_span, "parent"):
                return curr_span.parent is None
        except Exception as e:
            logger.warning(f"Error finding root span: {e}")

    def is_non_workflow_root_span(self, curr_span: Span, to_wrap) -> bool:
        return SpanHandler.is_root_span(curr_span) and to_wrap.get("span_type") != "workflow"
    
    def is_workflow_span_active(self):
        return get_value(WORKFLOW_TYPE_KEY) is not None

    @staticmethod
    def attach_workflow_type(to_wrap=None, context=None): 
        token = None
        if to_wrap:
            if to_wrap.get('span_type') == "workflow":
                token = attach(set_value(WORKFLOW_TYPE_KEY,
                                        SpanHandler.get_workflow_type(to_wrap), context))
        else:
            token = attach(set_value(WORKFLOW_TYPE_KEY, WORKFLOW_TYPE_GENERIC, context))
        return token

    @staticmethod
    def detach_workflow_type(token):
        if token:
            return detach(token)

class NonFrameworkSpanHandler(SpanHandler):

    # If the language framework is being executed, then skip generating direct openAI spans
    def skip_span(self, to_wrap, wrapped, instance, args, kwargs) -> bool:
        return get_value(WORKFLOW_TYPE_KEY) in WORKFLOW_TYPE_MAP.values()