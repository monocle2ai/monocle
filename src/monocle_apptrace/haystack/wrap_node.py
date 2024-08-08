

import logging
from opentelemetry import context as context_api
from opentelemetry.context import attach, set_value
from opentelemetry.instrumentation.utils import (
    _SUPPRESS_INSTRUMENTATION_KEY,
)
from monocle_apptrace.wrap_common import WORKFLOW_TYPE_MAP, with_tracer_wrapper

logger = logging.getLogger(__name__)


@with_tracer_wrapper
def wrap(tracer, to_wrap, wrapped, instance, args, kwargs):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)
    name = instance.name
    attach(set_value("workflow_name", name))
    with tracer.start_as_current_span(f"{name}.task") as span:
        workflow_name = span.resource.attributes.get("service.name")
        span.set_attribute("workflow_name",workflow_name)
        span.set_attribute("workflow_type", WORKFLOW_TYPE_MAP["haystack"])

        response = wrapped(*args, **kwargs)

    return response
