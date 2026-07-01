"""
Entity definition for Monocle feedback spans.

Simple feedback entity for capturing user feedback from agents.
Requires: session_id, feedback (string)
Optional: turn_id
"""

from monocle_apptrace.instrumentation.common.utils import get_error_message
from monocle_apptrace.instrumentation.metamodel.feedback import _helper

FEEDBACK_TYPE = "monocle.feedback"

FEEDBACK = {
    "type": FEEDBACK_TYPE,
    "attributes": [
        [
            {
                "_comment": "Feedback type",
                "attribute": "type",
                "accessor": lambda arguments: FEEDBACK_TYPE
            }
            
        ]
    ],
    "events": [
        {
            "name": "data.output",
            "attributes": [
                {
                    "_comment": "User feedback as string",
                    "attribute": "feedback",
                    "accessor": lambda arguments: _helper.extract_feedback_string(arguments)
                },
                {
                    "attribute": "error_code",
                    "accessor": lambda arguments: get_error_message(arguments)
                }
            ]
        }
    ]
}
