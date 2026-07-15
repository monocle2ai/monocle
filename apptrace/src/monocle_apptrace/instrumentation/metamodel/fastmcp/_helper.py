import logging
import json

logger = logging.getLogger(__name__)


def get_url(arguments):
    """Get the URL of the tool/prompt/resource from the instance."""
    try:
        instance = arguments.get("instance")
        if not instance:
            return None
            
        if not hasattr(instance, 'settings'):
            logger.debug("Instance does not have settings attribute")
            return None
            
        settings = instance.settings
        
        if not all(hasattr(settings, attr) for attr in ['host', 'port']):
            logger.debug("Settings missing required attributes (host, port)")
            return None
        
        base_url = f"http://{settings.host}:{settings.port}"
        return base_url
        
    except AttributeError as e:
        logger.debug(f"Attribute error extracting URL: {e}")
    except Exception as e:
        logger.warning(f"Unexpected error extracting URL from FastMCP instance: {e}")
    
    return None

def get_server_name(arguments):
    """Get the server name from the instance."""
    try:
        instance = arguments.get("instance")
        if not instance:
            return None
            
        if not hasattr(instance, 'name'):
            logger.debug("Instance does not have name attribute")
            return None
            
        return instance.name
        
    except AttributeError as e:
        logger.debug(f"Attribute error extracting server name: {e}")
    except Exception as e:
        logger.warning(f"Unexpected error extracting server name from FastMCP instance: {e}")
    
    return None


def get_list_input(arguments):
    """Get input for list operations (usually empty or pagination cursor)."""
    try:
        args = arguments.get("args", [])
        if not args:
            return []
            
        first_arg = args[0]
        if first_arg is None:
            return []
            
        try:
            return [json.dumps({"args": first_arg})]
        except (TypeError, ValueError) as e:
            logger.debug(f"Could not serialize list input args: {e}")
            return [str(first_arg)]
            
    except Exception as e:
        logger.warning(f"Error getting list input: {e}")
        return []


def get_list_of_names(arguments):
    """Get the list of tool/resource/prompt names from the result."""
    try:
        result = arguments.get("result")
        if not result:
            return []
        
        # Ensure result is iterable
        if not hasattr(result, '__iter__'):
            logger.debug(f"Result is not iterable: {type(result)}")
            return []
        
        names = []
        for item in result:
            if not item:
                continue
                
            # Get name attribute
            if hasattr(item, 'name'):
                names.append(item.name)
            elif isinstance(item, dict) and 'name' in item:
                names.append(item['name'])
            else:
                logger.debug(f"Item has no 'name' attribute: {type(item)}")
                
        return names
        
    except Exception as e:
        logger.warning(f"Error getting list of names: {e}")
        return []


def get_instance_name(arguments):
    """Get the name of the tool/prompt/resource being invoked."""
    try:
        args = arguments.get("args", [])
        if not args:
            return None
            
        # First argument is typically the name
        return args[0] if args[0] is not None else None
        
    except Exception as e:
        logger.warning(f"Error getting instance name: {e}")
        return None


def get_params_arguments(arguments):
    """Get the parameters/arguments passed to the tool/prompt."""
    try:
        args = arguments.get("args", [])
        if not args or len(args) < 2:
            return []
        
        # Second argument contains the parameters
        params = args[1]
        if params is None:
            return []
            
        try:
            return [json.dumps(params)]
        except (TypeError, ValueError) as e:
            logger.debug(f"Could not serialize params: {e}")
            # Fallback to string representation
            try:
                return [str(params)]
            except Exception:
                return []
                
    except Exception as e:
        logger.warning(f"Error getting params arguments: {e}")
        return []


def get_list_output(arguments):
    """Get the list output from result (generic serialization)."""
    try:
        result = arguments.get("result")
        if not result:
            return []
        
        # Try to serialize to JSON for structured output
        try:
            # If result is a list, try to extract meaningful info
            if isinstance(result, list):
                items = []
                for item in result:
                    if hasattr(item, '__dict__'):
                        # Try to get a clean dict representation
                        items.append({k: v for k, v in item.__dict__.items() if not k.startswith('_')})
                    else:
                        items.append(str(item))
                return json.dumps(items) if items else str(result)
        except Exception:
            pass
            
        return str(result)
        
    except Exception as e:
        logger.warning(f"Error getting list output: {e}")
        return []


def get_tool_output(arguments):
    """Get the tool execution output from result."""
    try:
        result = arguments.get('result')
        if not result:
            return []
        
        # Handle tuple result format (content, metadata)
        if isinstance(result, tuple) and len(result) >= 2:
            # Extract from second element (metadata dict)
            metadata = result[1]
            if isinstance(metadata, dict) and 'result' in metadata:
                return str(metadata['result'])
        
        # Handle direct result
        if isinstance(result, dict) and 'result' in result:
            return str(result['result'])
            
        # Fallback to string representation
        return str(result) if result else []
        
    except Exception as e:
        logger.warning(f"Error getting tool output: {e}")
        return []


def get_resource_uri(arguments):
    """Get the resource URI from arguments."""
    try:
        args = arguments.get("args", [])
        if not args:
            return []
        
        # First argument is the resource URI
        uri = args[0]
        if uri is None:
            return []
            
        try:
            return [json.dumps({"uri": uri})]
        except (TypeError, ValueError) as e:
            logger.debug(f"Could not serialize resource URI: {e}")
            return [str(uri)]
            
    except Exception as e:
        logger.warning(f"Error getting resource URI: {e}")
        return []


def get_resource_output(arguments):
    """Get the resource content from the result."""
    try:
        result = arguments.get("result")
        if not result:
            return []
        
        # Handle list of ReadResourceContents
        if isinstance(result, list) and result:
            # Get content from first item
            first_item = result[0]
            if hasattr(first_item, 'content'):
                return str(first_item.content)
            return str(first_item)
        
        # Handle single ReadResourceContents object
        if hasattr(result, 'content'):
            return str(result.content)
            
        # Fallback
        return str(result)
        
    except Exception as e:
        logger.warning(f"Error getting resource output: {e}")
        return []


def get_prompt_output(arguments):
    """Get the prompt messages from the result."""
    try:
        result = arguments.get("result")
        if not result:
            return []
        
        # Try to extract message content
        if hasattr(result, 'messages') and result.messages:
            # Get first message
            first_message = result.messages[0]
            if hasattr(first_message, 'content'):
                content = first_message.content
                # Extract text if it's a TextContent object
                if hasattr(content, 'text'):
                    return str(content.text)
                return str(content)
        
        # Fallback to string representation
        return str(result)
        
    except Exception as e:
        logger.warning(f"Error getting prompt output: {e}")
        return []


