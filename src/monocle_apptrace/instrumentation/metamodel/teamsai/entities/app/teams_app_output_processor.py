import json
from monocle_apptrace.instrumentation.metamodel.teamsai import (
    _helper,
)

TEAMS_APP_OUTPUT_PROCESSOR = {
    "type": "application",
    "attributes": [
        [{
            "attribute": "request.method",
            "accessor": lambda arguments: _helper.capture_teams_request_info(arguments).get('method', '')
        },
        {
            "attribute": "request.path", 
            "accessor": lambda arguments: _helper.capture_teams_request_info(arguments).get('path', '')
        },
        {
            "attribute": "activity.type",
            "accessor": lambda arguments: _helper.capture_teams_activity_info(arguments).get('activity_type', '')
        },
        {
            "attribute": "activity.channel_id",
            "accessor": lambda arguments: _helper.capture_teams_activity_info(arguments).get('channel_id', '')
        },
        {
            "attribute": "conversation.id",
            "accessor": lambda arguments: _helper.capture_teams_activity_info(arguments).get('conversation_id', '')
        },
        {
            "attribute": "conversation.type",
            "accessor": lambda arguments: _helper.capture_teams_activity_info(arguments).get('conversation_type', '')
        },
        {
            "attribute": "user.id",
            "accessor": lambda arguments: _helper.capture_teams_activity_info(arguments).get('user_id', '')
        },
        {
            "attribute": "user.name",
            "accessor": lambda arguments: _helper.capture_teams_activity_info(arguments).get('user_name', '')
        }]
    ],
    "events": [
        {
            "name": "data.input",
            "_comment": "Input data for Teams Application process or AI run",
            "attributes": [
                {
                    # "attribute": "request_info",
                    "accessor": lambda arguments: _helper.capture_teams_request_info(arguments)
                },
                {
                    "attribute": "activity_text",
                    "accessor": lambda arguments: _helper.capture_teams_activity_info(arguments).get('activity_text', '')
                },
                {
                    # "attribute": "conversation_context",
                    "accessor": lambda arguments: {
                        **_helper.capture_teams_activity_info(arguments),
                        **_helper.capture_teams_state_info(arguments)
                    }
                },
                {
                    "attribute": "current_tasks",
                    "accessor": lambda arguments: _helper.capture_teams_state_info(arguments).get('current_tasks', '')
                }
            ]
        },
        {
            "name": "data.output",
            "_comment": "Output data from Teams Application process or AI run",
            "attributes": [
                {
                    # "attribute": "response_info",
                    "accessor": lambda arguments: _helper.capture_teams_response_info(arguments)
                },
                {
                    "attribute": "status",
                    "accessor": lambda arguments: _helper.capture_teams_response_info(arguments).get('status', '')
                },
                {
                    "attribute": "result_summary",
                    "accessor": lambda arguments: _helper.capture_teams_response_info(arguments).get('result_summary', '')
                }
            ]
        },
        {
            "name": "metadata",
            "attributes": [
                {
                    # "attribute": "component_info",
                    "accessor": lambda arguments: _helper.capture_teams_metadata(arguments)
                },
                {
                    "attribute": "processing_type",
                    "accessor": lambda arguments: _helper.capture_teams_metadata(arguments).get('processing_type', '')
                },
                {
                    # "attribute": "timing_info",
                    "accessor": lambda arguments: _helper.capture_teams_timing_info(arguments)
                }
            ]
        }
    ]
}