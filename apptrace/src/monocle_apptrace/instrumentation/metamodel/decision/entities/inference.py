"""Shared decision-model INFERENCE output processor.

`make_decision_inference(adapter)` returns the output_processor dict. The span shape
lives here once; each decision SDK injects its own extraction via a `DecisionAdapter`.
Span type is reused `SPAN_TYPES.INFERENCE`. The model block carries `model_type`
from the adapter (completion when unset); decision models are tagged
`model.decision.<name>`, completion models `model.llm.<name>`.
"""

from monocle_apptrace.instrumentation.common.constants import MODEL_TYPE_ATTRIBUTE, SPAN_TYPES
from monocle_apptrace.instrumentation.common.utils import get_error_message
from monocle_apptrace.instrumentation.metamodel.decision import _helper


def make_decision_inference(adapter: "_helper.DecisionAdapter") -> dict:
    return {
        "type": SPAN_TYPES.INFERENCE,
        "attributes": [
            [
                {
                    "_comment": "decision provider type",
                    "attribute": "type",
                    "accessor": lambda arguments: "inference." + adapter.provider_name,
                },
                {
                    "attribute": "provider_name",
                    "accessor": lambda arguments: adapter.provider_name,
                },
                {
                    "attribute": "inference_endpoint",
                    "accessor": lambda arguments: adapter.get_inference_endpoint(
                        arguments["instance"]
                    ),
                },
            ],
            [
                {
                    "_comment": "resolved decision model name",
                    "attribute": "name",
                    "phase": "post_execution",
                    "accessor": lambda arguments: adapter.get_model(arguments),
                },
                {
                    "attribute": "type",
                    "phase": "post_execution",
                    "accessor": lambda arguments: _helper.model_entity_type(
                        adapter, adapter.get_model(arguments)
                    ),
                },
                {
                    "_comment": "decision or completion",
                    "attribute": MODEL_TYPE_ATTRIBUTE,
                    "phase": "post_execution",
                    "accessor": lambda arguments: _helper.model_type(adapter),
                },
            ],
        ],
        "events": [
            {
                "name": "data.input",
                "attributes": [
                    {
                        "_comment": "state + questions sent to the decision model",
                        "attribute": "input",
                        "accessor": lambda arguments: adapter.extract_input(arguments),
                    }
                ],
            },
            {
                "name": "data.output",
                "attributes": [
                    {
                        "attribute": "error_code",
                        "accessor": lambda arguments: get_error_message(arguments),
                    },
                    {
                        "_comment": "answers from the decision model",
                        "attribute": "response",
                        "accessor": lambda arguments: adapter.extract_answers(arguments),
                    },
                ],
            },
            {
                "name": "metadata",
                "attributes": [
                    {
                        "_comment": "token usage",
                        "accessor": lambda arguments: _helper.usage_metadata(
                            adapter.get_usage(arguments)
                        ),
                    },
                    {
                        "attribute": "finish_reason",
                        "accessor": lambda arguments: _helper.decision_finish_reason(arguments),
                    },
                    {
                        "attribute": "finish_type",
                        "accessor": lambda arguments: _helper.decision_finish_type(arguments),
                    },
                ],
            },
        ],
    }
