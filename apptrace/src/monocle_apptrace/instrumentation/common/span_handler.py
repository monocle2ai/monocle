import logging
import os
from contextlib import contextmanager
from opentelemetry.context import get_value, set_value, attach, detach
from opentelemetry.sdk.trace import Span
from opentelemetry.trace.status import Status, StatusCode
from monocle_apptrace.instrumentation.common.constants import (
    QUERY,
    service_name_map,
    service_type_map,
    MONOCLE_SDK_VERSION, MONOCLE_SDK_LANGUAGE, MONOCLE_DETECTED_SPAN_ERROR
)
from monocle_apptrace.instrumentation.common.utils import set_attribute, get_scopes, MonocleSpanException, get_monocle_version
from monocle_apptrace.instrumentation.common.constants import WORKFLOW_TYPE_KEY, WORKFLOW_TYPE_GENERIC, CHILD_ERROR_CODE

logger = logging.getLogger(__name__)

WORKFLOW_TYPE_MAP = {
    "llama_index.core.agent.workflow": WORKFLOW_TYPE_GENERIC,
    "llama_index": "workflow.llamaindex",
    "langchain": "workflow.langchain",
    "haystack": "workflow.haystack",
    "teams.ai": "workflow.teams_ai",
    "langgraph": "workflow.langgraph",
    "openai": "workflow.openai",
    "anthropic": "workflow.anthropic",
    "gemini": "workflow.gemini",
    "litellm": "workflow.litellm",
    "mistralai": "workflow.mistral",
    "huggingface_hub": "workflow.huggingface"
}

FRAMEWORK_WORKFLOW_LIST = [
    "workflow.llamaindex",
    "workflow.langchain",
    "workflow.haystack",
    "workflow.teams_ai",
    "workflow.litellm",
]
class SpanHandler:

    def __init__(self,instrumentor=None):
        self.instrumentor=instrumentor

    def set_instrumentor(self, instrumentor):
        self.instrumentor = instrumentor

    def validate(self, to_wrap, wrapped, instance, args, kwargs):
        pass

    def pre_tracing(self, to_wrap, wrapped, instance, args, kwargs):
        pass

    def post_tracing(self, to_wrap, wrapped, instance, args, kwargs, return_value, token=None):
        pass

    def skip_span(self, to_wrap, wrapped, instance, args, kwargs) -> bool:
        return False

    def skip_processor(self, to_wrap, wrapped, instance, span, args, kwargs) -> list[str]:
        return []
    
    def set_span_type(self, to_wrap, wrapped, instance, output_processor, span:Span, args, kwargs) -> str:
        span_type:str = None
        if 'type' in output_processor:
            span_type = output_processor['type']
            span.set_attribute("span.type", span_type)
        else:
            logger.warning("type of span not found or incorrect written in entity json")
        if "subtype" in output_processor:
            span.set_attribute("span.subtype", output_processor["subtype"])
        return span_type

    def pre_task_processing(self, to_wrap, wrapped, instance, args,kwargs, span):
        try:
            if "pipeline" in to_wrap['package']:
                set_attribute(QUERY, args[0]['prompt_builder']['question'])
        except Exception as e:
            logger.warning("Warning: Error occurred in pre_task_processing: %s", str(e))

    @staticmethod
    def set_default_monocle_attributes(span: Span, source_path = "" ):
        """ Set default monocle attributes for all spans """
        span.set_attribute(MONOCLE_SDK_VERSION, get_monocle_version())
        span.set_attribute(MONOCLE_SDK_LANGUAGE, "python")
        span.set_attribute("span_source", source_path)
        for scope_key, scope_value in get_scopes().items():
            span.set_attribute(f"scope.{scope_key}", scope_value)
        workflow_name = SpanHandler.get_workflow_name(span=span)
        if workflow_name:
            span.set_attribute("workflow.name", workflow_name)

    @staticmethod
    def set_workflow_properties(span: Span, to_wrap = None):
        """ Set attributes of workflow if this is a root span"""
        SpanHandler.set_workflow_attributes(to_wrap, span)
        SpanHandler.set_app_hosting_identifier_attribute(span)

    @staticmethod
    def set_non_workflow_properties(span: Span, to_wrap = None):
        span.set_attribute("span.type", "generic")

    def post_task_processing(self, to_wrap, wrapped, instance, args, kwargs, result, ex, span:Span, parent_span:Span):
        pass

    def should_skip(self, processor, instance, args, kwargs) -> bool:
        should_skip = False
        accessor = processor.get('should_skip')
        if accessor:
            arguments = {"instance":instance, "args":args, "kwargs":kwargs}
            should_skip = accessor(arguments)
            if not isinstance(should_skip, bool):
                logger.warning("Warning: 'should_skip' accessor did not return a boolean value")
                return True
        return should_skip

    def hydrate_span(self, to_wrap, wrapped, instance, args, kwargs, result, span, parent_span = None, ex:Exception = None) -> bool:
        try:
            detected_error_in_attribute = self.hydrate_attributes(to_wrap, wrapped, instance, args, kwargs, result, span, parent_span)
            detected_error_in_event = self.hydrate_events(to_wrap, wrapped, instance, args, kwargs, result, span, parent_span, ex)
            if detected_error_in_attribute or detected_error_in_event:
                span.set_attribute(MONOCLE_DETECTED_SPAN_ERROR, True)
        finally:
            if span.status.status_code == StatusCode.UNSET and ex is None:
                span.set_status(StatusCode.OK)

    def hydrate_attributes(self, to_wrap, wrapped, instance, args, kwargs, result, span:Span, parent_span:Span) -> bool:
        detected_error:bool = False
        span_index = 0
        if SpanHandler.is_root_span(span):
            span_index = 2 # root span will have workflow and hosting entities pre-populated
        if 'output_processor' in to_wrap and to_wrap["output_processor"] is not None:
            output_processor=to_wrap['output_processor']
            self.set_span_type(to_wrap, wrapped, instance, output_processor, span, args, kwargs)
            skip_processors:list[str] = self.skip_processor(to_wrap, wrapped, instance, span, args, kwargs) or []

            if 'attributes' in output_processor and 'attributes' not in skip_processors:
                arguments = {"instance":instance, "args":args, "kwargs":kwargs, "result":result, "parent_span":parent_span, "span":span}
                subtype = output_processor.get('subtype')
                if subtype:
                    try:
                        span.subtype_result = subtype(arguments)
                        span.set_attribute("span.subtype", span.subtype_result)
                    except Exception as e:
                        logger.debug(f"Error processing subtype: {e}")
                for processors in output_processor["attributes"]:
                    for processor in processors:
                        attribute = processor.get('attribute')
                        accessor = processor.get('accessor')

                        if attribute and accessor:
                            attribute_name = f"entity.{span_index+1}.{attribute}"
                            try:
                                processor_result = accessor(arguments)
                                if processor_result and isinstance(processor_result, (str, list)):
                                    span.set_attribute(attribute_name, processor_result)
                            except MonocleSpanException as e:
                                span.set_status(StatusCode.ERROR, e.message)
                                detected_error = True
                            except Exception as e:
                                logger.debug(f"Error processing accessor: {e}")
                        else:
                            logger.debug(f"{' and '.join([key for key in ['attribute', 'accessor'] if not processor.get(key)])} not found or incorrect in entity JSON")
                    span_index += 1

        # set scopes as attributes by calling get_scopes()
        # scopes is a Mapping[str:object], iterate directly with .items()
        for scope_key, scope_value in get_scopes().items():
            span.set_attribute(f"scope.{scope_key}", scope_value)
        
        if span_index > 0:
            span.set_attribute("entity.count", span_index)
        return detected_error

    def hydrate_events(self, to_wrap, wrapped, instance, args, kwargs, ret_result, span: Span, parent_span=None, ex:Exception=None) -> bool:
        detected_error:bool = False
        if 'output_processor' in to_wrap and to_wrap["output_processor"] is not None:
            output_processor=to_wrap['output_processor']
            skip_processors:list[str] = self.skip_processor(to_wrap, wrapped, instance, span, args, kwargs) or []

            arguments = {"instance": instance, "args": args, "kwargs": kwargs, "result": ret_result, "exception":ex, "parent_span":parent_span, "span": span}
            # Process events if they are defined in the output_processor.
            # In case of inference.modelapi skip the event processing unless the span has an exception
            if 'events' in output_processor and ('events' not in skip_processors or ex is not None):
                events = output_processor['events']
                for event in events:
                    event_name = event.get("name")
                    if 'events.'+event_name in skip_processors and ex is None:
                        continue
                    event_attributes = {}
                    attributes = event.get("attributes", [])
                    for attribute in attributes:
                        attribute_key = attribute.get("attribute")
                        accessor = attribute.get("accessor")
                        if accessor:
                            try:
                                try:
                                    result = accessor(arguments)
                                except MonocleSpanException as e:
                                    span.set_status(StatusCode.ERROR, e.message)
                                    detected_error = True
                                    result = e.get_err_code()
                                if result and isinstance(result, dict):
                                    result = dict((key, value) for key, value in result.items() if value is not None)
                                if result and isinstance(result, (int, str, list, dict)):
                                    if attribute_key is not None:
                                        event_attributes[attribute_key] = result
                                    else:
                                        event_attributes.update(result)
                            except Exception as e:
                                logger.debug(f"Error evaluating accessor for attribute '{attribute_key}': {e}")
                    matching_timestamp = getattr(ret_result, "timestamps", {}).get(event_name, None)
                    alreadyExist = False
                    for existing_event in span.events:
                        if event_name == existing_event.name:
                            existing_event.attributes._dict.update(event_attributes)
                            alreadyExist = True
                    if not alreadyExist:
                        if isinstance(matching_timestamp, int):
                            span.add_event(name=event_name, attributes=event_attributes, timestamp=matching_timestamp)
                        else:
                            span.add_event(name=event_name, attributes=event_attributes)
        return detected_error

    @staticmethod
    def set_workflow_attributes(to_wrap, span: Span):
        span_index = 1
        workflow_name = SpanHandler.get_workflow_name(span=span)
        if workflow_name:
            span.update_name("workflow")
            span.set_attribute("span.type", "workflow")
            span.set_attribute(f"entity.{span_index}.name", workflow_name)
        workflow_type = SpanHandler.get_workflow_type(to_wrap)
        span.set_attribute(f"entity.{span_index}.type", workflow_type)

    def get_workflow_name_in_progress(self) -> str:
        return get_value(WORKFLOW_TYPE_KEY)

    @staticmethod
    def is_framework_workflow(workflow_type) -> bool:
        return workflow_type in FRAMEWORK_WORKFLOW_LIST

    def is_framework_span_in_progress(self) -> bool:
        return SpanHandler.is_framework_workflow(self.get_workflow_name_in_progress())

    @staticmethod
    def get_workflow_type(to_wrap):
        # workflow type
        workflow_type = WORKFLOW_TYPE_GENERIC
        if to_wrap is not None:
            package_name = to_wrap.get('package')
            for (package, framework_workflow_type) in WORKFLOW_TYPE_MAP.items():
                if (package_name is not None and package_name.startswith(package)):
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
            if curr_span is not None and hasattr(curr_span, "parent") or  curr_span.context.trace_state:
                return curr_span.parent is None
        except Exception as e:
            logger.warning(f"Error finding root span: {e}")

    @staticmethod
    def attach_workflow_type(to_wrap=None, context=None): 
        token = None
        if to_wrap:
            workflow_type = SpanHandler.get_workflow_type(to_wrap)
            if SpanHandler.is_framework_workflow(workflow_type):
                token = attach(set_value(WORKFLOW_TYPE_KEY,
                                        SpanHandler.get_workflow_type(to_wrap), context))
        return token

    @staticmethod
    def detach_workflow_type(token):
        if token:
            return detach(token)

    @staticmethod
    @contextmanager
    def workflow_type(to_wrap=None, span:Span=None):
        token = SpanHandler.attach_workflow_type(to_wrap)
        try:
            yield
        finally:
            SpanHandler.detach_workflow_type(token)


class NonFrameworkSpanHandler(SpanHandler):
    # If the language framework is being executed, then skip generating direct openAI attributes and events
    def skip_processor(self, to_wrap, wrapped, instance, span, args, kwargs) -> list[str]:
        if super().is_framework_span_in_progress():
            return ["attributes", "events"]
    
    def set_span_type(self, to_wrap, wrapped, instance, output_processor, span:Span, args, kwargs) -> str:
        span_type = super().set_span_type(to_wrap, wrapped, instance, output_processor, span, args, kwargs)
        if self.is_framework_span_in_progress() and span_type is not None:
            span_type = span_type+".modelapi"
            span.set_attribute("span.type", span_type)
        return span_type

