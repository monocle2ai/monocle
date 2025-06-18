import logging
from monocle_apptrace.instrumentation.common.utils import MonocleSpanException
from monocle_apptrace.instrumentation.common.utils import (
    Option,
    get_keys_as_tuple,
    get_nested_value,
    try_option,
    get_exception_message,
    get_exception_status_code
)

logger = logging.getLogger(__name__)

def log_out(arguments):
    # log output
    print(f"Debug - Arguments received: {str(arguments)}")
    return

def capture_input(arguments):
    """
    Captures the input from Teams AI state.
    Args:
        arguments (dict): Arguments containing state and context information
    Returns:
        str: The input message or error message
    """
    try:
        # Get the memory object from kwargs
        kwargs = arguments.get("kwargs", {})
        messages = []

        # If memory exists, try to get the input from temp
        if "memory" in kwargs:
            memory = kwargs["memory"]
            # Check if it's a TurnState object
            if hasattr(memory, "get"):
                # Use proper TurnState.get() method
                temp = memory.get("temp")
                if temp and hasattr(temp, "get"):
                    input_value = temp.get("input")
                    if input_value:
                        messages.append({'user': str(input_value)})
        system_prompt = ""
        try:
            system_prompt = kwargs.get("template").prompt.sections[0].sections[0].template
            messages.append({'system': system_prompt})
        except Exception as e:
            logger.debug(f"Debug - Error accessing system prompt: {str(e)}")
            
        # Try alternative path through context if memory path fails
        context = kwargs.get("context")
        if hasattr(context, "activity") and hasattr(context.activity, "text"):
            messages.append({'user': str(context.activity.text)})

        return [str(message) for message in messages]
    except Exception as e:
        logger.debug(f"Debug - Arguments structure: {str(arguments)}")
        logger.debug(f"Debug - kwargs: {str(kwargs)}")
        if "memory" in kwargs:
            logger.debug(f"Debug - memory type: {type(kwargs['memory'])}")
        return f"Error capturing input: {str(e)}"

def capture_prompt_info(arguments):
    """Captures prompt information from ActionPlanner state"""
    try:
        kwargs = arguments.get("kwargs", {})
        prompt = kwargs.get("prompt")

        if isinstance(prompt, str):
            return prompt
        elif hasattr(prompt, "name"):
            return prompt.name

        return "No prompt information found"
    except Exception as e:
        return f"Error capturing prompt: {str(e)}"

def capture_prompt_template_info(arguments):
    """Captures prompt information from ActionPlanner state"""
    try:
        kwargs = arguments.get("kwargs", {})
        prompt = kwargs.get("prompt")

        if hasattr(prompt,"prompt") and prompt.prompt is not None:
            if "_text" in prompt.prompt.__dict__:
                prompt_template = prompt.prompt.__dict__.get("_text", None)
                return prompt_template
            elif "_sections" in prompt.prompt.__dict__ and prompt.prompt._sections is not None:
                sections = prompt.prompt._sections[0].__dict__.get("_sections", None)
                if sections is not None and "_template" in sections[0].__dict__:
                    return sections[0].__dict__.get("_template", None)


        return "No prompt information found"
    except Exception as e:
        return f"Error capturing prompt: {str(e)}"

def status_check(arguments):
    if hasattr(arguments["result"], "error") and arguments["result"].error is not None:
        error_msg:str = arguments["result"].error
        error_code:str = arguments["result"].status if hasattr(arguments["result"], "status") else "unknown"
        raise MonocleSpanException(f"Error: {error_code} - {error_msg}")

def get_prompt_template(arguments):
    pass
    return {
        "prompt_template_name": capture_prompt_info(arguments),
        "prompt_template": capture_prompt_template_info(arguments),
        "prompt_template_description": get_nested_value(arguments.get("kwargs", {}), ["prompt", "config", "description"]),
        "prompt_template_type": get_nested_value(arguments.get("kwargs", {}), ["prompt", "config", "type"])
    }

def get_status_code(arguments):
    if arguments["exception"] is not None:
        return get_exception_status_code(arguments)
    elif hasattr(arguments["result"], "status"):
        return arguments["result"].status
    else:
        return 'success'

def get_status(arguments):
    if arguments["exception"] is not None:
        return 'error'
    elif get_status_code(arguments) == 'success':
        return 'success'
    else:
        return 'error'
    
def get_response(arguments) -> str:
    status = get_status_code(arguments)
    response:str = ""
    if status == 'success':
        if hasattr(arguments["result"], "message"):
            response = arguments["result"].message.content 
        else:
            response = str(arguments["result"])
    else:
        if arguments["exception"] is not None:
            response = get_exception_message(arguments)
        elif hasattr(arguments["result"], "error"):
            response = arguments["result"].error
    return response

def check_status(arguments):
    status = get_status_code(arguments)
    if status != 'success':
        raise MonocleSpanException(f"{status}")   

def extract_provider_name(instance):
    provider_url: Option[str] = try_option(getattr, instance._client.base_url, 'host')
    return provider_url.unwrap_or(None)


def extract_inference_endpoint(instance):
    inference_endpoint: Option[str] = try_option(getattr, instance._client, 'base_url').map(str)
    if inference_endpoint.is_none() and "meta" in instance.client.__dict__:
        inference_endpoint = try_option(getattr, instance.client.meta, 'endpoint_url').map(str)

    return inference_endpoint.unwrap_or(extract_provider_name(instance))

def capture_teams_request_info(arguments):
    """Captures request information from Teams Application.process method"""
    try:
        args = arguments.get("args", ())
        kwargs = arguments.get("kwargs", {})
        
        request_info = {}
        
        # Extract request object (first argument for Application.process)
        if args and len(args) > 0:
            request = args[0]
            if hasattr(request, 'method'):
                request_info['method'] = request.method
            if hasattr(request, 'path'):
                request_info['path'] = request.path
            if hasattr(request, 'headers'):
                # Get content type if available
                content_type = request.headers.get('content-type', '')
                if content_type:
                    request_info['content_type'] = content_type
        
        return request_info
    except Exception as e:
        return {"error": f"Error capturing request info: {str(e)}"}

def capture_teams_activity_info(arguments):
    """Captures activity and context information from Teams methods"""
    try:
        activity_info = {}
        
        # Check for context in kwargs (common in AI.run method)
        context  = arguments.get("args", {})[0]
        if context and hasattr(context, 'activity'):
            activity = context.activity
            
            activity_info['activity_type'] = getattr(activity, 'type', '')
            activity_info['activity_text'] = getattr(activity, 'text', '')
            activity_info['channel_id'] = getattr(activity, 'channel_id', '')
            
            # Conversation details
            if hasattr(activity, 'conversation'):
                conv = activity.conversation
                activity_info['conversation_id'] = getattr(conv, 'id', '')
                activity_info['conversation_type'] = getattr(conv, 'conversation_type', '')
                activity_info['conversation_name'] = getattr(conv, 'name', '')
            
            # User information
            if hasattr(activity, 'from_property'):
                user = activity.from_property
                activity_info['user_id'] = getattr(user, 'id', '')
                activity_info['user_name'] = getattr(user, 'name', '')
                activity_info['user_role'] = getattr(user, 'role', '')
            
            # Recipient information
            if hasattr(activity, 'recipient'):
                recipient = activity.recipient
                activity_info['recipient_id'] = getattr(recipient, 'id', '')
        
        return activity_info
    except Exception as e:
        return {"error": f"Error capturing activity info: {str(e)}"}

def capture_teams_state_info(arguments):
    """Captures state information from Teams AI methods"""
    try:
        kwargs = arguments.get("kwargs", {})
        state_info = {}
        
        # Check for state in kwargs (common in AI.run method)
        state = kwargs.get("state")
        if state:
            # Try to get conversation tasks if available
            if hasattr(state, 'conversation') and hasattr(state.conversation, 'tasks'):
                tasks = state.conversation.tasks
                if tasks:
                    state_info['current_tasks'] = str(tasks)
                    state_info['task_count'] = len(tasks) if isinstance(tasks, dict) else 0
            
            # Get memory/temp information if available
            if hasattr(state, 'temp'):
                temp = state.temp
                if hasattr(temp, 'input'):
                    state_info['input_text'] = str(temp.input)
        
        return state_info
    except Exception as e:
        return {"error": f"Error capturing state info: {str(e)}"}

def capture_teams_response_info(arguments):
    """Captures response information from Teams methods"""
    try:
        result = arguments.get("result")
        exception = arguments.get("exception")
        
        response_info = {}
        
        if exception:
            response_info['status'] = 'error'
            response_info['error_message'] = str(exception)
        else:
            response_info['status'] = 'success'
            
            # For Application.process, result might be a Response object
            if result and hasattr(result, 'status'):
                response_info['http_status'] = result.status
            
            # For AI.run, result is typically a boolean
            if isinstance(result, bool):
                response_info['ai_run_result'] = result
            
            # Try to capture any string representation
            if result is not None:
                response_info['result_summary'] = str(result)[:200]  # Truncate long results
        
        return response_info
    except Exception as e:
        return {"error": f"Error capturing response info: {str(e)}"}

def capture_teams_timing_info(arguments):
    """Captures timing and performance information"""
    try:
        timing_info = {}
        
        # Check if timing information is available in arguments
        start_time = arguments.get("start_time")
        end_time = arguments.get("end_time")
        
        if start_time and end_time:
            duration = (end_time - start_time).total_seconds()
            timing_info['duration_seconds'] = duration
        
        return timing_info
    except Exception as e:
        return {"error": f"Error capturing timing info: {str(e)}"}

def capture_teams_metadata(arguments):
    """Captures metadata about the Teams application and processing"""
    try:
        metadata = {}
        
        # Method-specific metadata
        method_name = arguments.get("method_name", "")
        if method_name:
            metadata['method'] = method_name
        
        # Try to determine if this is Application.process or AI.run
        if "process" in method_name:
            metadata['component'] = 'teams_application'
        elif "run" in method_name:
            metadata['component'] = 'teams_ai'
        
        # Add processing type
        metadata['processing_type'] = 'teams_bot_framework'
        
        return metadata
    except Exception as e:
        return {"error": f"Error capturing metadata: {str(e)}"}

def capture_conversation_state_context_info(arguments):
    """Captures context information from conversation state loading"""
    try:
        # Get the context from args (first argument in load method)
        args = arguments.get("args", ())
        context_info = {}
        
        if args and len(args) > 1:  # cls is first, context is second
            context = args[1]
            if hasattr(context, 'activity'):
                activity = context.activity
                
                context_info['channel_id'] = getattr(activity, 'channel_id', '')
                context_info['activity_type'] = getattr(activity, 'type', '')
                
                # Conversation details
                if hasattr(activity, 'conversation'):
                    conv = activity.conversation
                    context_info['conversation_id'] = getattr(conv, 'id', '')
                    context_info['conversation_type'] = getattr(conv, 'conversation_type', '')
                    context_info['conversation_name'] = getattr(conv, 'name', '')
                
                # Bot details
                if hasattr(activity, 'recipient'):
                    recipient = activity.recipient
                    context_info['bot_id'] = getattr(recipient, 'id', '')
                    context_info['bot_name'] = getattr(recipient, 'name', '')
                
                # User details
                if hasattr(activity, 'from_property'):
                    user = activity.from_property
                    context_info['user_id'] = getattr(user, 'id', '')
                    context_info['user_name'] = getattr(user, 'name', '')
        
        return context_info
    except Exception as e:
        return {"error": f"Error capturing context info: {str(e)}"}

def capture_conversation_state_storage_info(arguments):
    """Captures storage information from conversation state loading"""
    try:
        args = arguments.get("args", ())
        storage_info = {}
        
        if args and len(args) > 2:  # cls, context, storage
            storage = args[2]
            if storage:
                storage_info['storage_type'] = type(storage).__name__
                storage_info['has_storage'] = True
            else:
                storage_info['has_storage'] = False
        else:
            storage_info['has_storage'] = False
        
        return storage_info
    except Exception as e:
        return {"error": f"Error capturing storage info: {str(e)}"}

def capture_conversation_state_key_info(arguments):
    """Captures the generated state key information"""
    try:
        args = arguments.get("args", ())
        key_info = {}
        
        if args and len(args) > 1:
            context = args[1]
            if hasattr(context, 'activity'):
                activity = context.activity
                
                # Reconstruct the key that would be generated
                channel_id = getattr(activity, 'channel_id', '')
                conversation_id = ''
                bot_id = ''
                
                if hasattr(activity, 'conversation'):
                    conversation_id = getattr(activity.conversation, 'id', '')
                
                if hasattr(activity, 'recipient'):
                    bot_id = getattr(activity.recipient, 'id', '')
                
                if channel_id and conversation_id and bot_id:
                    key = f"{channel_id}/{bot_id}/conversations/{conversation_id}"
                    key_info['state_key'] = key
                    key_info['key_components'] = {
                        'channel_id': channel_id,
                        'bot_id': bot_id,
                        'conversation_id': conversation_id
                    }
        
        return key_info
    except Exception as e:
        return {"error": f"Error capturing key info: {str(e)}"}

def capture_conversation_state_load_result(arguments):
    """Captures information about the loaded conversation state"""
    try:
        result = arguments.get("result")
        exception = arguments.get("exception")
        result_info = {}
        
        if exception:
            result_info['status'] = 'error'
            result_info['error_message'] = str(exception)
        else:
            result_info['status'] = 'success'
            
            if result:
                result_info['state_type'] = type(result).__name__
                
                # Check if the result has a __key__ attribute
                if hasattr(result, '__key__'):
                    result_info['loaded_key'] = getattr(result, '__key__', '')
                
                # Check if it's AppConversationState with tasks
                if hasattr(result, 'tasks'):
                    tasks = result.tasks
                    if tasks:
                        result_info['has_tasks'] = True
                        result_info['task_count'] = len(tasks) if isinstance(tasks, dict) else 0
                        result_info['task_keys'] = list(tasks.keys()) if isinstance(tasks, dict) else []
                    else:
                        result_info['has_tasks'] = False
                        result_info['task_count'] = 0
                
                # Check for other state attributes
                state_attrs = []
                for attr in dir(result):
                    if not attr.startswith('_') and not callable(getattr(result, attr)):
                        state_attrs.append(attr)
                result_info['state_attributes'] = state_attrs
        
        return result_info
    except Exception as e:
        return {"error": f"Error capturing load result: {str(e)}"}

def capture_conversation_state_metadata(arguments):
    """Captures metadata about conversation state loading operation"""
    try:
        metadata = {}
        
        # Determine the class being instantiated
        args = arguments.get("args", ())
        if args and len(args) > 0:
            cls = args[1]
            if hasattr(cls, '__name__'):
                metadata['state_class'] = cls.__name__
            else:
                metadata['state_class'] = str(type(cls))
        
        # Operation type
        metadata['operation'] = 'conversation_state_load'
        metadata['component'] = 'teams_conversation_state'
        
        # Check if this is the base ConversationState or a derived class
        if args and hasattr(args[0], '__name__'):
            if args[0].__name__ == 'ConversationState':
                metadata['is_base_class'] = True
            else:
                metadata['is_base_class'] = False
                metadata['derived_class'] = args[0].__name__
        
        return metadata
    except Exception as e:
        return {"error": f"Error capturing metadata: {str(e)}"}
def extract_search_endpoint(instance):
    if hasattr(instance, '_endpoint') and instance._endpoint is not None:
        return instance._endpoint
    else:
        return None

def extract_index_name(instance):
    if hasattr(instance, '_index_name') and instance._index_name is not None:
        return instance._index_name
    else:
        return None

def capture_vector_queries(kwargs):
    try:
        if 'vector_queries' in kwargs and kwargs['vector_queries'] is not None:
            vector_queries = kwargs['vector_queries'][0]
            if hasattr(vector_queries, 'vector') and len(vector_queries.vector) > 0:
                return vector_queries.vector[:10]
    except Exception as e:
        print(f"Debug - Error capturing vector queries: {str(e)}")

def search_input(arguments):
    pass
    return {
        "search_text": get_nested_value(arguments.get("kwargs", {}), ["search_text"]),
        "select": get_nested_value(arguments.get("kwargs", {}), ["select"]),
        "vector_queries": capture_vector_queries(arguments["kwargs"])
    }

def search_output(arguments):
    try:
        if hasattr(arguments["result"], "_args") and len(arguments["result"]._args) > 1:
            if hasattr(arguments["result"]._args[1], "request") and arguments["result"]._args[1].request is not None:
                request = arguments["result"]._args[1].request
                return {
                    "count": request.include_total_result_count,
                    "coverage": request.minimum_coverage,
                    "facets": request.facets,
                    }
    except Exception as e:
        print(f"Debug - Error capturing facets: {str(e)}")
    return None

