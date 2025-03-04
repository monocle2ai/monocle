import logging
import os
from importlib.metadata import version
from opentelemetry.context import get_current
from opentelemetry.context import get_value
from opentelemetry.sdk.trace import Span

from monocle_apptrace.instrumentation.common.constants import (
    QUERY,
    service_name_map,
    service_type_map,
)
from monocle_apptrace.instrumentation.common.utils import set_attribute, get_scopes

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
        return False

    def pre_task_processing(self, to_wrap, wrapped, instance, args,kwargs, span):
        if self.__is_root_span(span):
            try:
                sdk_version = version("monocle_apptrace")
                span.set_attribute("monocle_apptrace.version", sdk_version)
            except Exception as e:
                logger.warning("Exception finding monocle-apptrace version.")
        if "pipeline" in to_wrap['package']:
            set_attribute(QUERY, args[0]['prompt_builder']['question'])

    def post_task_processing(self, to_wrap, wrapped, instance, args, kwargs, result, span):
        pass

    def hydrate_span(self, to_wrap, wrapped, instance, args, kwargs, result, span):
        self.hydrate_attributes(to_wrap, wrapped, instance, args, kwargs, result, span)
        self.hydrate_events(to_wrap, wrapped, instance, args, kwargs, result, span)

    def hydrate_attributes(self, to_wrap, wrapped, instance, args, kwargs, result, span):
        span_index = 0
        if self.__is_root_span(span):
            span_index += self.set_workflow_attributes(to_wrap, span, span_index+1)
            span_index += self.set_app_hosting_identifier_attribute(span, span_index+1)

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
                            except Exception as e:
                                logger.debug(f"Error evaluating accessor for attribute '{attribute_key}': {e}")
                    span.add_event(name=event_name, attributes=event_attributes)



    def set_workflow_attributes(self, to_wrap, span: Span, span_index):
        return_value = 1
        workflow_name = self.get_workflow_name(span=span)
        if workflow_name:
            span.set_attribute("span.type", "workflow")
            span.set_attribute(f"entity.{span_index}.name", workflow_name)
            # workflow type
        package_name = to_wrap.get('package')
        workflow_type_set = False
        for (package, workflow_type) in WORKFLOW_TYPE_MAP.items():
            if (package_name is not None and package in package_name):
                span.set_attribute(f"entity.{span_index}.type", workflow_type)
                workflow_type_set = True
        if not workflow_type_set:
            span.set_attribute(f"entity.{span_index}.type", "workflow.generic")
        return return_value

    def set_app_hosting_identifier_attribute(self, span, span_index):
        return_value = 0
        # Search env to indentify the infra service type, if found check env for service name if possible
        for type_env, type_name in service_type_map.items():
            if type_env in os.environ:
                return_value = 1
                span.set_attribute(f"entity.{span_index}.type", f"app_hosting.{type_name}")
                entity_name_env = service_name_map.get(type_name, "unknown")
                span.set_attribute(f"entity.{span_index}.name", os.environ.get(entity_name_env, "generic"))
        return return_value

    def get_workflow_name(self, span: Span) -> str:
        try:
            return get_value("workflow_name") or span.resource.attributes.get("service.name")
        except Exception as e:
            logger.exception(f"Error getting workflow name: {e}")
            return None

    def __is_root_span(self, curr_span: Span) -> bool:
        try:
            if curr_span is not None and hasattr(curr_span, "parent"):
                return curr_span.parent is None or get_current().get("root_span_id") == curr_span.parent.span_id
        except Exception as e:
            logger.warning(f"Error finding root span: {e}")
