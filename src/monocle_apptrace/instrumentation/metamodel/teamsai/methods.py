from monocle_apptrace.instrumentation.common.wrapper import (
    ascopes_wrapper,
    atask_wrapper,
    task_wrapper,
    ascope_wrapper,
)
from monocle_apptrace.instrumentation.metamodel.teamsai.entities.inference.teamsai_output_processor import (
    TEAMAI_OUTPUT_PROCESSOR,
)
from monocle_apptrace.instrumentation.metamodel.teamsai.entities.inference.actionplanner_output_processor import (
    ACTIONPLANNER_OUTPUT_PROCESSOR,
)


def get_id(args, kwargs):
    """
    Extracts the ID from the context.
    """
    scopes: dict[str, dict[str:str]] = {}
    context = kwargs.get("context")
    if context and context.activity and context.activity.conversation.id:
        conversation_id = context.activity.conversation.id or ""
        user_aad_object_id = context.activity.from_property.aad_object_id or ""
        user_teams_id = context.activity.from_property.id or ""
        channel_id = context.activity.channel_id or ""
        recipient_id = context.activity.recipient.id or ""
        recipient_aad_object_id = context.activity.recipient.aad_object_id or ""
        scopes[f"teams.conversation.conversation.id"] = conversation_id
        scopes[f"teams.user.from_property.aad_object_id"] = user_aad_object_id
        scopes[f"teams.user.from_property.id"] = user_teams_id
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
    },
    {
        "package": "teams.ai.planners.action_planner",
        "object": "ActionPlanner",
        "method": "complete_prompt",
        "wrapper_method": atask_wrapper,
        "output_processor": ACTIONPLANNER_OUTPUT_PROCESSOR,
    },
    {
        "package": "teams.ai.planners.action_planner",
        "object": "ActionPlanner",
        "method": "complete_prompt",
        "scope_values": get_id,
        "wrapper_method": ascopes_wrapper,
    },
]
