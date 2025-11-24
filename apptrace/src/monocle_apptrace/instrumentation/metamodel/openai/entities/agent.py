from monocle_apptrace.instrumentation.common.constants import SPAN_TYPES, SPAN_SUBTYPES
from monocle_apptrace.instrumentation.metamodel.openai import _helper
from monocle_apptrace.instrumentation.common.utils import get_error_message


AGENT = {
    "type": SPAN_TYPES.AGENTIC_INVOCATION,
    "subtype": SPAN_SUBTYPES.CONTENT_PROCESSING,
    "attributes": [
        [
            {
                "_comment": "agent type",
                "attribute": "type",
                "accessor": lambda arguments: "agent.openai"
            },
            {
                "_comment": "name of the agent",
                "attribute": "name",
                "accessor": lambda arguments: getattr(arguments.get('args', [None])[0], 'name', 'unknown_agent') if arguments.get('args') else 'unknown_agent'
            }
        ]
    ],
    "events": [
        {
            "name": "data.input",
            "attributes": [
                {
                    "_comment": "this is Agent input",
                    "attribute": "input",
                    "accessor": lambda arguments: [arguments.get('kwargs', {}).get('input', '')]
                }
            ]
        },
        {
            "name": "data.output",
            "attributes": [
                {
                    "attribute": "error_code",
                    "accessor": lambda arguments: get_error_message(arguments)
                },
                {
                    "_comment": "this is response from Agent",
                    "attribute": "response",
                    "accessor": lambda arguments: str(arguments.get('result', ''))
                }
            ]
        }
    ]
}
