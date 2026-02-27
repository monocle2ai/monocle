"""Processor handlers for Microsoft Agent Framework instrumentation."""

import json
import logging
from opentelemetry.context import attach, set_value, detach, get_value
from monocle_apptrace.instrumentation.common.span_handler import SpanHandler
from monocle_apptrace.instrumentation.common.constants import AGENT_INVOCATION_SPAN_NAME, AGENT_NAME_KEY, AGENT_SESSION, INFERENCE_AGENT_DELEGATION, INFERENCE_TURN_END, LAST_AGENT_INVOCATION_ID, LAST_AGENT_NAME, SPAN_TYPES, INFERENCE_TOOL_CALL, AGENT_PREFIX_KEY
from monocle_apptrace.instrumentation.common.utils import set_scope, remove_scope
from monocle_apptrace.instrumentation.common.utils import get_exception_message, get_json_dumps, get_status_code
from opentelemetry.trace import Span

logger = logging.getLogger(__name__)

# Context key for storing agent information
MSAGENT_CONTEXT_KEY = "msagent.agent_info"

def propogate_agent_name_to_parent_span(span: Span, parent_span: Span):
    """Propagate agent name from child span to parent span."""
    if span.attributes.get("span.type") != AGENT_INVOCATION_SPAN_NAME:
        return
    if parent_span is not None:
        parent_span.set_attribute(LAST_AGENT_INVOCATION_ID, hex(span.context.span_id))
        # Try to get agent name from context first, then fall back to span attributes
        agent_name = get_value(AGENT_NAME_KEY)
        if agent_name is None:
            # Context may have been detached, try reading from span attributes
            agent_name = span.attributes.get("entity.1.name")
        if agent_name is not None:
            parent_span.set_attribute(LAST_AGENT_NAME, agent_name)

class MSAgentRequestHandler(SpanHandler):
    """Handler for Microsoft Agent Framework turn-level requests (agentic.request)."""
    
    def skip_span(self, to_wrap, wrapped, instance, args, kwargs):
        """Skip ChatAgent.run span when it's the 2nd+ turn span (inside workflow/multi-agent)."""
        from monocle_apptrace.instrumentation.common.utils import is_scope_set
        
        # Skip if turn scope is already set - means we're the 2nd+ span
        # The first span (workflow or first agent) already created the turn
        if is_scope_set("agentic.turn"):
            logger.debug(f"Skipping ChatAgent.run span - turn scope already set (inside workflow/multi-agent)")
            return True
        return False
    
    def pre_tracing(self, to_wrap, wrapped, instance, args, kwargs):
        """Called before turn execution to extract and store agent information in context."""
        from monocle_apptrace.instrumentation.metamodel.msagent._helper import uses_chat_client, is_inside_workflow
        from monocle_apptrace.instrumentation.metamodel.msagent.entities.inference import AGENT, AGENT_REQUEST
        from monocle_apptrace.instrumentation.common.utils import is_scope_set, set_scopes
        from monocle_apptrace.instrumentation.common.scope_wrapper import start_scope
        
        # Store agent information in context for child spans to access
        agent_info = {}
        if hasattr(instance, "name"):
            agent_info["name"] = instance.name
        if hasattr(instance, "instructions"):
            agent_info["instructions"] = instance.instructions
        
        # Extract thread/session ID and set scope
        session_id_token = None
        thread = kwargs.get("thread")
        if thread is not None:
            # Extract thread ID from thread object
            thread_id = None
            if hasattr(thread, "service_thread_id"):
                thread_id = thread.service_thread_id
            elif hasattr(thread, "id"):
                thread_id = thread.id
            elif hasattr(thread, "thread_id"):
                thread_id = thread.thread_id
            
            if thread_id:
                session_id_token = set_scope(AGENT_SESSION, thread_id)
        
        # Attach agent info context
        context_token = None
        if agent_info:
            context_token = attach(set_value(MSAGENT_CONTEXT_KEY, agent_info))
        
        # Extract MS Teams context and set scopes
        msteams_token = None
        scope_values = self._extract_msteams_context(args, kwargs)
        if scope_values:
            msteams_token = set_scopes(scope_values)
            
            # Filter out context parameters from kwargs to prevent passing them to underlying methods
            filtered_kwargs = {k: v for k, v in kwargs.items() 
                               if k not in ["context", "turn_context", "turnContext"]}
            
            # Update kwargs in place to filter out context parameters
            kwargs.clear()
            kwargs.update(filtered_kwargs)
        
        # Store session token for cleanup (can't return it with context_token)
        self._session_token = session_id_token
        
        # Store MS Teams token for cleanup
        self._msteams_token = msteams_token
        
        # Determine processor based on client type and context
        scope_name = AGENT_REQUEST.get("type")
        alternate_to_wrap = None
        
        if uses_chat_client(instance):
            # ChatClient: use recursive processor list to create turn + invocation
            logger.debug(f"ChatClient: setting output_processor_list=[AGENT_REQUEST, AGENT]")
            alternate_to_wrap = to_wrap.copy()
            alternate_to_wrap["output_processor_list"] = [AGENT_REQUEST, AGENT]
            # Clear output_processor if it exists to avoid confusion
            if "output_processor" in alternate_to_wrap:
                del alternate_to_wrap["output_processor"]
        else:
            # AssistantsClient: ChatAgent.run creates turn only, AssistantsClient methods create invocation
            if not is_scope_set(scope_name):
                # Create turn scope, AssistantsClient will create invocation later
                logger.debug(f"AssistantsClient: setting output_processor=AGENT_REQUEST")
                alternate_to_wrap = to_wrap.copy()
                alternate_to_wrap["output_processor"] = AGENT_REQUEST
        
        return context_token, alternate_to_wrap
    
    def post_tracing(self, to_wrap, wrapped, instance, args, kwargs, result, token):
        """Clean up tokens set in pre_tracing."""
        # Detach context token passed as parameter
        if token is not None:
            detach(token)
        
        # Clean up session token if it exists
        if hasattr(self, '_session_token') and self._session_token is not None:
            remove_scope(self._session_token)
            self._session_token = None
        
        # Clean up MS Teams token if it exists
        if hasattr(self, '_msteams_token') and self._msteams_token is not None:
            remove_scope(self._msteams_token)
            self._msteams_token = None
    
    def _extract_msteams_context(self, args, kwargs):
        """Extract MS Teams context from Bot Framework TurnContext."""
        scopes = {}
        
        # Try to find TurnContext in various possible locations
        context = None
        
        # Check kwargs for common parameter names
        for param_name in ["context", "turn_context", "turnContext"]:
            if param_name in kwargs:
                context = kwargs[param_name]
                break
        
        # Check args if not found in kwargs
        if not context and len(args) > 0:
            for arg in args:
                # Check if this looks like a TurnContext object
                if hasattr(arg, 'activity') and hasattr(arg.activity, 'channel_id'):
                    context = arg
                    break
        
        if not context or not hasattr(context, 'activity'):
            return scopes
        
        activity = context.activity
        
        # Extract channel information
        if hasattr(activity, 'channel_id') and activity.channel_id:
            channel_id = activity.channel_id
            scopes["teams.channel.channel_id"] = channel_id
            
            # If it's MS Teams, extract Teams-specific attributes
            if channel_id == "msteams":
                # Activity type
                if hasattr(activity, 'type') and activity.type:
                    scopes["msteams.activity.type"] = activity.type
                
                # Conversation information
                if hasattr(activity, "conversation") and activity.conversation:
                    if hasattr(activity.conversation, 'id'):
                        scopes["msteams.conversation.id"] = activity.conversation.id or ""
                    if hasattr(activity.conversation, 'conversation_type'):
                        scopes["msteams.conversation.type"] = activity.conversation.conversation_type or ""
                    if hasattr(activity.conversation, 'name'):
                        scopes["msteams.conversation.name"] = activity.conversation.name or ""
                
                # User information (from_property)
                if hasattr(activity, "from_property") and activity.from_property:
                    if hasattr(activity.from_property, 'id'):
                        scopes["msteams.user.from_property.id"] = activity.from_property.id or ""
                    if hasattr(activity.from_property, 'name'):
                        scopes["msteams.user.from_property.name"] = activity.from_property.name or ""
                    if hasattr(activity.from_property, 'role'):
                        scopes["msteams.user.from_property.role"] = activity.from_property.role or ""
                
                # Recipient information
                if hasattr(activity, "recipient") and activity.recipient:
                    if hasattr(activity.recipient, 'id'):
                        scopes["msteams.recipient.id"] = activity.recipient.id or ""
                
                # Channel data (tenant, team, channel details)
                if hasattr(activity, "channel_data") and activity.channel_data:
                    channel_data = activity.channel_data
                    
                    # Tenant information
                    if isinstance(channel_data, dict):
                        if "tenant" in channel_data and "id" in channel_data["tenant"]:
                            scopes["msteams.channel_data.tenant.id"] = channel_data["tenant"]["id"] or ""
                        
                        # Team information
                        if "team" in channel_data:
                            if "id" in channel_data["team"]:
                                scopes["msteams.channel_data.team.id"] = channel_data["team"]["id"] or ""
                            if "name" in channel_data["team"]:
                                scopes["msteams.channel_data.team.name"] = channel_data["team"]["name"] or ""
                        
                        # Channel information
                        if "channel" in channel_data:
                            if "id" in channel_data["channel"]:
                                scopes["msteams.channel_data.channel.id"] = channel_data["channel"]["id"] or ""
                            if "name" in channel_data["channel"]:
                                scopes["msteams.channel_data.channel.name"] = channel_data["channel"]["name"] or ""
        
        return scopes

    def post_task_processing(self, to_wrap, wrapped, instance, args, kwargs, result, ex, span: Span, parent_span: Span):
        self.hydrate_events(to_wrap, wrapped, instance, args, kwargs,
                            result, span=parent_span, is_post_exec=True)

class MSAgentAgentHandler(SpanHandler):
    """Handler for Microsoft Agent Framework agent invocations."""
    
    def skip_span(self, to_wrap, wrapped, instance, args, kwargs):
        """Skip get_response/get_streaming_response for ChatClient only in standalone mode."""
        from monocle_apptrace.instrumentation.common.utils import is_scope_set
        
        client_class = instance.__class__.__name__
        # For ChatClient: skip only in standalone mode (when no turn scope exists yet)
        # In workflow/multi-agent: turn scope exists, so get_response creates invocation
        if client_class == "AzureOpenAIChatClient":
            if is_scope_set("agentic.turn"):
                # Turn scope exists - we're in workflow/multi-agent, don't skip
                logger.debug(f"Not skipping get_response - turn scope exists (workflow), create invocation span")
                return False
            else:
                # No turn scope - standalone mode where ChatAgent.run creates both via processor_list
                logger.debug(f"Skipping get_response - standalone mode, ChatAgent.run creates via processor_list")
                return True
        # Don't skip for AssistantsClient - this is where invocation span is created
        return False
    
    def pre_tracing(self, to_wrap, wrapped, instance, args, kwargs):
        """Set agent name in context with duplicate-invocation guard for ChatClient."""
        from monocle_apptrace.instrumentation.common.utils import is_scope_set
        
        # For ChatClient: only create span if we're in recursive output_processor_list flow
        # This happens when ChatAgent.run uses output_processor_list=[AGENT_REQUEST, AGENT]
        # The second recursive call needs get_response to create the invocation span
        client_class = instance.__class__.__name__
        if client_class == "AzureOpenAIChatClient":
            # Skip only if invocation scope already exists (would be duplicate)
            if is_scope_set("agentic.invocation"):
                return None, None  # Skip span creation
        
        # Set agent name in context for propagation
        agent_name = None
        if hasattr(instance, "name"):
            agent_name = instance.name
        elif hasattr(instance, "_name"):
            agent_name = instance._name
        if agent_name:
            context = set_value(AGENT_NAME_KEY, agent_name)
            token = attach(context)
            return token, None
        return None, None

    def post_tracing(self, to_wrap, wrapped, instance, args, kwargs, result, token):
        """Called after agent execution to clean up context."""
        self._context_token = token  # Store for later cleanup

    def post_task_processing(self, to_wrap, wrapped, instance, args, kwargs, result, ex, span, parent_span):
        """Propagate agent name and invocation ID to parent span, then clean up context."""
        # Propagate while context still has agent name
        propogate_agent_name_to_parent_span(span, parent_span)
        # Now detach context
        if hasattr(self, '_context_token') and self._context_token is not None:
            detach(self._context_token)
            self._context_token = None
        return super().post_task_processing(to_wrap, wrapped, instance, args, kwargs, result, ex, span, parent_span)

    def hydrate_span(self, to_wrap, wrapped, instance, args, kwargs, result, span, parent_span=None, ex: Exception = None, is_post_exec: bool = False) -> bool:
        """Hydrate span with agent-specific attributes."""
        return super().hydrate_span(to_wrap, wrapped, instance, args, kwargs, result, span, parent_span, ex, is_post_exec)


class MSAgentInferenceHandler(SpanHandler):
    """Handler for Microsoft Agent Framework inference spans."""

    @staticmethod
    def _get_field(value, key, default=None):
        if value is None:
            return default
        if isinstance(value, dict):
            return value.get(key, default)
        return getattr(value, key, default)

    @staticmethod
    def _as_list(value):
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        return [value]

    @staticmethod
    def _extract_usage_tokens(usage_obj):
        if usage_obj is None:
            return {}

        completion_tokens = (
            MSAgentInferenceHandler._get_field(usage_obj, "output_token_count")
            or MSAgentInferenceHandler._get_field(usage_obj, "completion_tokens")
            or MSAgentInferenceHandler._get_field(usage_obj, "output_tokens")
        )
        prompt_tokens = (
            MSAgentInferenceHandler._get_field(usage_obj, "input_token_count")
            or MSAgentInferenceHandler._get_field(usage_obj, "prompt_tokens")
            or MSAgentInferenceHandler._get_field(usage_obj, "input_tokens")
        )
        total_tokens = (
            MSAgentInferenceHandler._get_field(usage_obj, "total_token_count")
            or MSAgentInferenceHandler._get_field(usage_obj, "total_tokens")
        )

        token_map = {}
        if completion_tokens is not None:
            token_map["completion_tokens"] = completion_tokens
        if prompt_tokens is not None:
            token_map["prompt_tokens"] = prompt_tokens
        if total_tokens is not None:
            token_map["total_tokens"] = total_tokens
        return token_map

    @staticmethod
    def _response_contains_tool_calls(response):
        try:
            if response is None:
                return False

            if MSAgentInferenceHandler._as_list(MSAgentInferenceHandler._get_field(response, "tools")):
                return True

            for output_item in MSAgentInferenceHandler._as_list(MSAgentInferenceHandler._get_field(response, "output")):
                item_type = MSAgentInferenceHandler._get_field(output_item, "type")
                if item_type in ("function_call", "tool_call", "tool_calls"):
                    return True
                if MSAgentInferenceHandler._as_list(MSAgentInferenceHandler._get_field(output_item, "tool_calls")):
                    return True
                if MSAgentInferenceHandler._get_field(output_item, "call_id") and MSAgentInferenceHandler._get_field(output_item, "name"):
                    return True

            required_action = MSAgentInferenceHandler._get_field(response, "required_action")
            submit_tool_outputs = MSAgentInferenceHandler._get_field(required_action, "submit_tool_outputs")
            if MSAgentInferenceHandler._as_list(MSAgentInferenceHandler._get_field(submit_tool_outputs, "tool_calls")):
                return True

            for message in MSAgentInferenceHandler._as_list(MSAgentInferenceHandler._get_field(response, "messages")):
                if MSAgentInferenceHandler._as_list(MSAgentInferenceHandler._get_field(message, "tool_calls")):
                    return True
                for content in MSAgentInferenceHandler._as_list(MSAgentInferenceHandler._get_field(message, "contents")):
                    content_type = MSAgentInferenceHandler._get_field(content, "type") or type(content).__name__
                    if content_type in ("FunctionCallContent", "function_call", "tool_call", "tool_calls"):
                        return True
                    if MSAgentInferenceHandler._get_field(content, "call_id") and MSAgentInferenceHandler._get_field(content, "name"):
                        return True

            for choice in MSAgentInferenceHandler._as_list(MSAgentInferenceHandler._get_field(response, "choices")):
                finish_reason = MSAgentInferenceHandler._get_field(choice, "finish_reason")
                if finish_reason in ("tool_calls", "function_call"):
                    return True
                if MSAgentInferenceHandler._as_list(MSAgentInferenceHandler._get_field(MSAgentInferenceHandler._get_field(choice, "message"), "tool_calls")):
                    return True
                if MSAgentInferenceHandler._as_list(MSAgentInferenceHandler._get_field(MSAgentInferenceHandler._get_field(choice, "delta"), "tool_calls")):
                    return True
        except Exception as exc:
            logger.debug(f"Error while checking tool call response: {exc}")
        return False

    @staticmethod
    def _extract_first_tool_name(response):
        try:
            for tool in MSAgentInferenceHandler._as_list(MSAgentInferenceHandler._get_field(response, "tools")):
                tool_name = MSAgentInferenceHandler._get_field(tool, "name")
                if tool_name:
                    return str(tool_name)

            for output_item in MSAgentInferenceHandler._as_list(MSAgentInferenceHandler._get_field(response, "output")):
                tool_name = MSAgentInferenceHandler._get_field(output_item, "name")
                if tool_name:
                    return str(tool_name)

            required_action = MSAgentInferenceHandler._get_field(response, "required_action")
            submit_tool_outputs = MSAgentInferenceHandler._get_field(required_action, "submit_tool_outputs")
            for tool_call in MSAgentInferenceHandler._as_list(MSAgentInferenceHandler._get_field(submit_tool_outputs, "tool_calls")):
                function_obj = MSAgentInferenceHandler._get_field(tool_call, "function")
                tool_name = MSAgentInferenceHandler._get_field(function_obj, "name") or MSAgentInferenceHandler._get_field(tool_call, "name")
                if tool_name:
                    return str(tool_name)

            for message in MSAgentInferenceHandler._as_list(MSAgentInferenceHandler._get_field(response, "messages")):
                for content in MSAgentInferenceHandler._as_list(MSAgentInferenceHandler._get_field(message, "contents")):
                    tool_name = MSAgentInferenceHandler._get_field(content, "name")
                    if tool_name:
                        return str(tool_name)

            for choice in MSAgentInferenceHandler._as_list(MSAgentInferenceHandler._get_field(response, "choices")):
                for tool_call in MSAgentInferenceHandler._as_list(MSAgentInferenceHandler._get_field(MSAgentInferenceHandler._get_field(choice, "message"), "tool_calls")):
                    function_obj = MSAgentInferenceHandler._get_field(tool_call, "function")
                    tool_name = MSAgentInferenceHandler._get_field(function_obj, "name") or MSAgentInferenceHandler._get_field(tool_call, "name")
                    if tool_name:
                        return str(tool_name)
        except Exception as exc:
            logger.debug(f"Error while extracting first tool name: {exc}")
        return None

    @staticmethod
    def extract_assistant_message(arguments):
        try:
            messages = []
            status = get_status_code(arguments)

            if status not in ("success", "completed"):
                if arguments.get("exception") is not None:
                    return get_exception_message(arguments)
                if hasattr(arguments.get("result"), "error"):
                    return arguments["result"].error
                return None

            response = arguments.get("result")
            if response is None:
                return ""

            if hasattr(response, "tools") and isinstance(response.tools, list) and response.tools:
                if isinstance(response.tools[0], dict):
                    tools = [
                        {
                            "tool_id": tool.get("id", ""),
                            "tool_name": tool.get("name", ""),
                            "tool_arguments": tool.get("arguments", ""),
                        }
                        for tool in response.tools
                    ]
                    messages.append({"tools": tools})

            if hasattr(response, "text") and response.text:
                messages.append({"assistant": response.text})

            if hasattr(response, "messages") and response.messages:
                for msg in MSAgentInferenceHandler._as_list(response.messages):
                    if hasattr(msg, "contents") and msg.contents:
                        tools = []
                        text_parts = []
                        for content in msg.contents:
                            content_type = type(content).__name__
                            if content_type == "FunctionCallContent" or (
                                hasattr(content, "call_id") and hasattr(content, "name")
                            ):
                                tools.append(
                                    {
                                        "tool_id": getattr(content, "call_id", ""),
                                        "tool_name": getattr(content, "name", ""),
                                        "tool_arguments": getattr(content, "arguments", ""),
                                    }
                                )
                            elif hasattr(content, "text") and content.text:
                                text_parts.append(content.text)
                            elif hasattr(content, "value") and content.value:
                                text_parts.append(content.value)

                        if tools:
                            messages.append({"tools": tools})
                        if text_parts:
                            messages.append({"assistant": " ".join(text_parts)})
                    elif hasattr(msg, "text") and msg.text:
                        messages.append({"assistant": msg.text})
                    elif hasattr(msg, "content") and msg.content:
                        messages.append({"assistant": msg.content})

            if hasattr(response, "output") and isinstance(response.output, list) and response.output:
                output_tools = []
                for output_item in response.output:
                    if getattr(output_item, "type", None) == "function_call":
                        output_tools.append(
                            {
                                "tool_id": getattr(output_item, "call_id", ""),
                                "tool_name": getattr(output_item, "name", ""),
                                "tool_arguments": getattr(output_item, "arguments", ""),
                            }
                        )
                if output_tools:
                    messages.append({"tools": output_tools})

            if hasattr(response, "output_text") and response.output_text:
                role = response.role if hasattr(response, "role") else "assistant"
                messages.append({role: response.output_text})

            if hasattr(response, "choices") and response.choices:
                first_choice = response.choices[0]
                if hasattr(first_choice, "message"):
                    role = getattr(first_choice.message, "role", "assistant")
                    messages.append({role: first_choice.message.content})

            return get_json_dumps(messages[0]) if messages else ""

        except (IndexError, AttributeError) as exc:
            logger.warning("Warning: Error occurred in extract_assistant_message: %s", str(exc))
            return None

    @staticmethod
    def agent_inference_type(arguments):
        try:
            response = arguments.get("result")
            if not MSAgentInferenceHandler._response_contains_tool_calls(response):
                return INFERENCE_TURN_END

            agent_prefix = get_value(AGENT_PREFIX_KEY)
            tool_name = MSAgentInferenceHandler._extract_first_tool_name(response) or ""
            if tool_name and agent_prefix and tool_name.startswith(agent_prefix):
                return INFERENCE_AGENT_DELEGATION
            return INFERENCE_TOOL_CALL
        except Exception as exc:
            logger.warning("Warning: Error occurred in agent_inference_type: %s", str(exc))
            return INFERENCE_TURN_END

    @staticmethod
    def extract_finish_reason(arguments):
        try:
            if arguments.get("exception") is not None and hasattr(arguments["exception"], "code"):
                return arguments["exception"].code

            response = arguments.get("result")
            if not response:
                return None

            if MSAgentInferenceHandler._response_contains_tool_calls(response):
                return "tool_calls"

            direct_finish_reason = MSAgentInferenceHandler._get_field(response, "finish_reason")
            if direct_finish_reason:
                return direct_finish_reason.value if hasattr(direct_finish_reason, "value") else str(direct_finish_reason)

            choices = MSAgentInferenceHandler._as_list(MSAgentInferenceHandler._get_field(response, "choices"))
            if choices:
                choice_finish_reason = MSAgentInferenceHandler._get_field(choices[0], "finish_reason")
                if choice_finish_reason:
                    return (
                        choice_finish_reason.value
                        if hasattr(choice_finish_reason, "value")
                        else str(choice_finish_reason)
                    )

            if hasattr(response, "text") or hasattr(response, "messages"):
                return "stop"

        except Exception as exc:
            logger.warning(f"Error extracting finish_reason: {exc}")
        return None

    @staticmethod
    def extract_tool_name(arguments):
        return MSAgentInferenceHandler._extract_first_tool_name(arguments.get("result"))

    @staticmethod
    def extract_tool_type(arguments):
        try:
            tool_name = MSAgentInferenceHandler.extract_tool_name(arguments)
            if not tool_name:
                return None
            agent_prefix = get_value(AGENT_PREFIX_KEY)
            if agent_prefix and tool_name.startswith(agent_prefix):
                return "agent.microsoft"
            return "tool.microsoft"
        except Exception as exc:
            logger.warning(f"Error extracting tool type: {exc}")
        return None

    @staticmethod
    def update_span_from_llm_response(response):
        meta_dict = {}
        try:
            if response is None:
                return meta_dict

            arguments = response if isinstance(response, dict) else {"result": response}
            result = arguments.get("result")

            if result is None:
                return meta_dict

            usage_candidates = [
                MSAgentInferenceHandler._get_field(result, "usage_details"),
                MSAgentInferenceHandler._get_field(result, "usage"),
                MSAgentInferenceHandler._get_field(
                    MSAgentInferenceHandler._get_field(result, "response_metadata"), "token_usage"
                ),
            ]

            choices = MSAgentInferenceHandler._as_list(MSAgentInferenceHandler._get_field(result, "choices"))
            if choices:
                usage_candidates.append(MSAgentInferenceHandler._get_field(choices[0], "usage"))

            messages = MSAgentInferenceHandler._as_list(MSAgentInferenceHandler._get_field(result, "messages"))
            for message in messages:
                usage_candidates.append(MSAgentInferenceHandler._get_field(message, "usage"))
                usage_candidates.append(
                    MSAgentInferenceHandler._get_field(
                        MSAgentInferenceHandler._get_field(message, "response_metadata"), "token_usage"
                    )
                )

            outputs = MSAgentInferenceHandler._as_list(MSAgentInferenceHandler._get_field(result, "output"))
            for output_item in outputs:
                usage_candidates.append(MSAgentInferenceHandler._get_field(output_item, "usage"))
                usage_candidates.append(
                    MSAgentInferenceHandler._get_field(
                        MSAgentInferenceHandler._get_field(output_item, "response_metadata"), "token_usage"
                    )
                )

            for usage_candidate in usage_candidates:
                meta_dict.update(MSAgentInferenceHandler._extract_usage_tokens(usage_candidate))
                if meta_dict:
                    return meta_dict

            if MSAgentInferenceHandler._response_contains_tool_calls(result):
                return {
                    "completion_tokens": 0,
                    "prompt_tokens": 0,
                    "total_tokens": 0,
                }

        except Exception as exc:
            logger.warning(f"Error updating span from LLM response: {exc}")
        return meta_dict

    def pre_tracing(self, to_wrap, wrapped, instance, args, kwargs):
        return None, None

    def post_tracing(self, to_wrap, wrapped, instance, args, kwargs, result, token):
        if token is not None:
            detach(token)


class MSAgentToolHandler(SpanHandler):
    """Handler for Microsoft Agent Framework tool invocations."""

    def pre_tracing(self, to_wrap, wrapped, instance, args, kwargs):
        """Called before tool execution to extract tool information."""
        return None, None

    def post_tracing(self, to_wrap, wrapped, instance, args, kwargs, result, token):
        """Called after tool execution to extract result information."""
        if token is not None:
            detach(token)

    def hydrate_span(self, to_wrap, wrapped, instance, args, kwargs, result, span, parent_span=None, ex: Exception = None, is_post_exec: bool = False) -> bool:
        """Hydrate span with tool-specific attributes."""
        return super().hydrate_span(to_wrap, wrapped, instance, args, kwargs, result, span, parent_span, ex, is_post_exec)

