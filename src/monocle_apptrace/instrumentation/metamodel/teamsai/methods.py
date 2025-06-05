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
    if context and context.activity and context.activity.channel_id:
        channel_id = context.activity.channel_id or ""
        scopes[f"teams.channel.channel_id"] = channel_id
        if channel_id == "msteams":
            scopes[f"msteams.activity.type"] = context.activity.type or ""

            if hasattr(context.activity,"conversation"):
                scopes[f"msteams.conversation.id"] = context.activity.conversation.id or ""
                scopes[f"msteams.conversation.type"] = context.activity.conversation.conversation_type or ""
                scopes[f"msteams.conversation.name"] = context.activity.conversation.name or ""

            if hasattr(context.activity,"from_property"):
                scopes[f"msteams.user.from_property.id"] = context.activity.from_property.id or ""
                scopes[f"msteams.user.from_property.name"] = context.activity.from_property.name or ""
                scopes[f"msteams.user.from_property.role"] = context.activity.from_property.role or ""

            if hasattr(context.activity,"recipient"):
                scopes[f"msteams.recipient.id"] = context.activity.recipient.id or ""

            if hasattr(context.activity,"channel_data"):
                if "tenant" in context.activity.channel_data:
                    scopes[f"msteams.channel_data.tenant.id"] = context.activity.channel_data['tenant']['id'] or ""
                if "team" in context.activity.channel_data:
                    scopes[f"msteams.channel_data.team.id"] = context.activity.channel_data['team']['id'] or ""
                    if "name" in context.activity.channel_data['team']:
                        scopes[f"msteams.channel_data.team.name"] = context.activity.channel_data['team']['name'] or ""
                if "channel" in context.activity.channel_data:
                    scopes[f"msteams.channel_data.channel.id"] = context.activity.channel_data['channel']['id'] or ""
                    if "name" in context.activity.channel_data['channel']:
                        scopes[f"msteams.channel_data.channel.name"] = context.activity.channel_data['channel']['name'] or ""
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
    }
]
