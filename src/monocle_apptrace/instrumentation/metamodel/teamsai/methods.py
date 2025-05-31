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
        if channel_id == "msteams":
            type_name = context.activity.type or ""
 
            if hasattr(context.activity,"conversation"):
                conversation_id = context.activity.conversation.id or ""
                converstion_type = context.activity.conversation.conversation_type or ""
                conversation_name = context.activity.conversation.name or ""
            if hasattr(context.activity,"from_property"):
                user_teams_id = context.activity.from_property.id or ""
                user_teams_name = context.activity.from_property.name or ""
                user_teams_role = context.activity.from_property.role or ""
            if hasattr(context.activity,"recipient"):
                recipient_id = context.activity.recipient.id or ""

            scopes[f"msteams.activity.type"] = type_name
            scopes[f"msteams.conversation.id"] = conversation_id
            scopes[f"msteams.conversation.type"] = converstion_type
            scopes[f"msteams.conversation.name"] = conversation_name
            scopes[f"msteams.user.from_property.id"] = user_teams_id
            scopes[f"msteams.user.from_property.name"] = user_teams_name
            scopes[f"msteams.user.from_property.role"] = user_teams_role
            scopes[f"msteams.recipient.id"] = recipient_id
            if hasattr(context.activity,"channel_data"):
                if "tenant" in context.activity.channel_data:
                    tenant_id = context.activity.channel_data['tenant']['id'] or ""
                    scopes[f"msteams.channel_data.tenant.id"] = tenant_id
                if "team" in context.activity.channel_data:
                    team_id = context.activity.channel_data['team']['id'] or ""
                    scopes[f"msteams.channel_data.team.id"] = team_id
                    if "name" in context.activity.channel_data['team']:
                        team_name = context.activity.channel_data['team']['name'] or ""
                        scopes[f"msteams.channel_data.team.name"] = team_name
                if "channel" in context.activity.channel_data:
                    team_channel_id = context.activity.channel_data['channel']['id'] or ""
                    scopes[f"msteams.channel_data.channel.id"] = team_channel_id
                    if "name" in context.activity.channel_data['channel']:
                        team_channel_name = context.activity.channel_data['channel']['name'] or ""
                        scopes[f"msteams.channel_data.channel.name"] = team_channel_name
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
