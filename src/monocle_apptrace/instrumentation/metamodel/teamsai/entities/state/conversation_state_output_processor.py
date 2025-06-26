import json
from monocle_apptrace.instrumentation.metamodel.teamsai import (
    _helper,
)

CONVERSATION_STATE_OUTPUT_PROCESSOR = {
    "type": "state_management",
    "attributes": [
        [{
            "attribute": "state.class",
            "accessor": lambda arguments: _helper.capture_conversation_state_metadata(arguments).get('state_class', '')
        },
        {
            "attribute": "state.key",
            "accessor": lambda arguments: _helper.capture_conversation_state_key_info(arguments).get('state_key', '')
        },
        {
            "attribute": "context.channel_id",
            "accessor": lambda arguments: _helper.capture_conversation_state_context_info(arguments).get('channel_id', '')
        },
        {
            "attribute": "context.conversation_id",
            "accessor": lambda arguments: _helper.capture_conversation_state_context_info(arguments).get('conversation_id', '')
        },
        {
            "attribute": "context.bot_id",
            "accessor": lambda arguments: _helper.capture_conversation_state_context_info(arguments).get('bot_id', '')
        },
        {
            "attribute": "context.user_id",
            "accessor": lambda arguments: _helper.capture_conversation_state_context_info(arguments).get('user_id', '')
        },
        {
            "attribute": "storage.has_storage",
            "accessor": lambda arguments: _helper.capture_conversation_state_storage_info(arguments).get('has_storage', False)
        },
        {
            "attribute": "storage.type",
            "accessor": lambda arguments: _helper.capture_conversation_state_storage_info(arguments).get('storage_type', '')
        }]
    ],
    "events": [
        {
            "name": "data.input",
            "_comment": "Input data for conversation state loading",
            "attributes": [
                {
                    # "attribute": "context_info",
                    "accessor": lambda arguments: _helper.capture_conversation_state_context_info(arguments)
                },
                {
                    # "attribute": "storage_info",
                    "accessor": lambda arguments: _helper.capture_conversation_state_storage_info(arguments)
                },
                {
                    # "attribute": "state_key_info",
                    "accessor": lambda arguments: _helper.capture_conversation_state_key_info(arguments)
                },
                {
                    "attribute": "conversation_id",
                    "accessor": lambda arguments: _helper.capture_conversation_state_context_info(arguments).get('conversation_id', '')
                },
                {
                    "attribute": "user_id",
                    "accessor": lambda arguments: _helper.capture_conversation_state_context_info(arguments).get('user_id', '')
                },
                {
                    "attribute": "channel_id",
                    "accessor": lambda arguments: _helper.capture_conversation_state_context_info(arguments).get('channel_id', '')
                }
            ]
        },
        {
            "name": "data.output",
            "_comment": "Output data from conversation state loading",
            "attributes": [
                {
                    # "attribute": "load_result",
                    "accessor": lambda arguments: _helper.capture_conversation_state_load_result(arguments)
                },
                {
                    "attribute": "status",
                    "accessor": lambda arguments: _helper.capture_conversation_state_load_result(arguments).get('status', '')
                },
                {
                    "attribute": "state_type",
                    "accessor": lambda arguments: _helper.capture_conversation_state_load_result(arguments).get('state_type', '')
                },
                {
                    "attribute": "state_response",
                    "accessor": lambda arguments: _helper.capture_conversation_state(arguments)
                },
            ]
        },
        {
            "name": "metadata",
            "attributes": [
                {
                    # "attribute": "operation_metadata",
                    "accessor": lambda arguments: _helper.capture_conversation_state_metadata(arguments)
                },
                {
                    "attribute": "operation_type",
                    "accessor": lambda arguments: _helper.capture_conversation_state_metadata(arguments).get('operation', '')
                },
                {
                    "attribute": "component",
                    "accessor": lambda arguments: _helper.capture_conversation_state_metadata(arguments).get('component', '')
                },
                {
                    "attribute": "is_base_class",
                    "accessor": lambda arguments: _helper.capture_conversation_state_metadata(arguments).get('is_base_class', True)
                },
                {
                    "attribute": "derived_class",
                    "accessor": lambda arguments: _helper.capture_conversation_state_metadata(arguments).get('derived_class', '')
                }
            ]
        }
    ]
}
