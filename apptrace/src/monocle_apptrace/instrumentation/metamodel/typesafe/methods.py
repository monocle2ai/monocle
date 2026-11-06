"""TypeSafe (jev) method instrumentation: system_one sync + async.

Both entries share the decision-model INFERENCE processor. No streaming path.
"""

from monocle_apptrace.instrumentation.common.wrapper import atask_wrapper, task_wrapper
from monocle_apptrace.instrumentation.metamodel.decision.entities.inference import (
    make_decision_inference,
)
from monocle_apptrace.instrumentation.metamodel.typesafe._helper import TYPESAFE_ADAPTER

DECISION_INFERENCE = make_decision_inference(TYPESAFE_ADAPTER)

TYPESAFE_METHODS = [
    {
        "package": "typesafe_sdk._core.client.sync.client",
        "object": "TypeSafeClient",
        "method": "system_one",
        "wrapper_method": task_wrapper,
        "output_processor": DECISION_INFERENCE,
        "span_handler": "non_framework_handler",
    },
    {
        "package": "typesafe_sdk._core.client.aio.client",
        "object": "AsyncTypeSafeClient",
        "method": "system_one",
        "wrapper_method": atask_wrapper,
        "output_processor": DECISION_INFERENCE,
        "span_handler": "non_framework_handler",
    },
]
