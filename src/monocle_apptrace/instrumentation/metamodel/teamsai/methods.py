from monocle_apptrace.instrumentation.common.wrapper import (
    ascopes_wrapper,
    atask_wrapper
)
from monocle_apptrace.instrumentation.metamodel.teamsai.entities.inference.teamsai_output_processor import (
    TEAMAI_OUTPUT_PROCESSOR,
)
from monocle_apptrace.instrumentation.metamodel.teamsai.entities.inference.actionplanner_output_processor import (
    ACTIONPLANNER_OUTPUT_PROCESSOR,
)

TEAMS_SCOPE_KEY = "teams.ai"

def get_id(args, kwargs, to_wrap):
    """
    Extracts the ID from the context.
    """
    scopes: dict[str, dict[str:str]] = {}
    context = kwargs.get("context")
    if context and context.activity and context.activity.conversation.id:
        conversation_id = context.activity.conversation.id or ""
        conversation_type = context.activity.conversation.conversation_type or ""
        scopes[f"teams.conversation.conversation.id"] = conversation_id
        scopes[f"teams.conversation.conversation.type"] = conversation_type
    if to_wrap and to_wrap.get("framework") and to_wrap["framework"] == TEAMS_SCOPE_KEY:
        required_scopes = to_wrap["scopes"] or []
        if "user" in required_scopes:
            user_aad_object_id = context.activity.from_property.aad_object_id or ""
            user_teams_id = context.activity.from_property.id or ""
            scopes[f"teams.user.from_property.aad_object_id"] = user_aad_object_id
            scopes[f"teams.user.from_property.id"] = user_teams_id

        if "channel" in required_scopes:
            channel_id = context.activity.channel_id or ""
            recipient_id = context.activity.recipient.id or ""
            recipient_aad_object_id = context.activity.recipient.aad_object_id or ""
            scopes[f"teams.channel.channel_id"] = channel_id
            scopes[f"teams.channel.recipient.id"] = recipient_id
            scopes[f"teams.channel.recipient.aad_object_id"] = recipient_aad_object_id

    return scopes


TEAMAI_METHODS = [
    {
        "package": "teams.ai.models.openai_model",
        "object": "OpenAIModel",
        "method": "complete_prompt",
        "wrapper_method": atask_wrapper,
        "output_processor": TEAMAI_OUTPUT_PROCESSOR,
        "framework": "teams.ai",
    },
    {
        "package": "teams.ai.planners.action_planner",
        "object": "ActionPlanner",
        "method": "complete_prompt",
        "wrapper_method": atask_wrapper,
        "output_processor": ACTIONPLANNER_OUTPUT_PROCESSOR,
        "framework": "teams.ai",
    },
    {
        "package": "teams.ai.planners.action_planner",
        "object": "ActionPlanner",
        "method": "complete_prompt",
        "scope_values": get_id,
        "wrapper_method": ascopes_wrapper,
        "framework": TEAMS_SCOPE_KEY,
    },
]
