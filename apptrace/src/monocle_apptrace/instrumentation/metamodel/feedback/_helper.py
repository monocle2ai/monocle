"""
Helper functions for extracting Monocle feedback data.

Simple helpers for extracting session_id, turn_id, and feedback string.
"""

import logging
from typing import Any, Dict, Optional
from monocle_apptrace.instrumentation.common.utils import get_status_code

logger = logging.getLogger(__name__)


def extract_session_id(arguments: Dict[str, Any]) -> Optional[str]:
    """
    Extract the session ID (required).
    
    Args:
        arguments: Dict containing instance, args, kwargs, result
        
    Returns:
        Session ID string
    """
    try:
        kwargs = arguments.get("kwargs", {})
        # Check kwargs first
        session_id = kwargs.get("session_id")
        if session_id:
            return str(session_id)
        
        # Check result
        result = arguments.get("result")
        if hasattr(result, "session_id"):
            return str(result.session_id)
        elif isinstance(result, dict) and result.get("session_id"):
            return str(result.get("session_id"))
        
        logger.warning("session_id is required but not found")
        return None
    except Exception as e:
        logger.warning(f"Error extracting session_id: {e}")
        return None


def extract_turn_id(arguments: Dict[str, Any]) -> Optional[str]:
    """
    Extract the turn ID (optional).
    
    Args:
        arguments: Dict containing instance, args, kwargs, result
        
    Returns:
        Turn ID string or None
    """
    try:
        kwargs = arguments.get("kwargs", {})
        # Check kwargs first
        turn_id = kwargs.get("turn_id")
        if turn_id:
            return str(turn_id)
        
        # Check result
        result = arguments.get("result")
        if hasattr(result, "turn_id"):
            return str(result.turn_id)
        elif isinstance(result, dict) and result.get("turn_id"):
            return str(result.get("turn_id"))
        
        return None
    except Exception as e:
        logger.warning(f"Error extracting turn_id: {e}")
        return None


def extract_feedback_string(arguments: Dict[str, Any]) -> Optional[str]:
    """
    Extract the feedback string value.
    
    Args:
        arguments: Dict containing instance, args, kwargs, result
        
    Returns:
        Feedback string
    """
    try:
        status = get_status_code(arguments)
        if status != 'success':
            return None
        
        result = arguments.get("result")
        
        # Check result object
        if hasattr(result, "feedback"):
            return str(result.feedback)
        elif isinstance(result, dict) and result.get("feedback"):
            return str(result.get("feedback"))
        elif isinstance(result, str):
            # Result itself is the feedback string
            return result
        
        # Check kwargs as fallback
        kwargs = arguments.get("kwargs", {})
        feedback = kwargs.get("feedback")
        if feedback:
            return str(feedback)
        
        logger.warning("feedback string not found in result or kwargs")
        return None
        
    except Exception as e:
        logger.warning(f"Error extracting feedback string: {e}")
        return None


# Legacy functions kept for backward compatibility but simplified
def extract_feedback_source(arguments: Dict[str, Any]) -> Optional[str]:
    """
    Extract the source/channel where feedback was collected.
    
    Args:
        arguments: Dict containing instance, args, kwargs, result
        
    Returns:
        Source identifier (e.g., 'web_chat', 'mobile_app', 'api')
    """
    try:
        kwargs = arguments.get("kwargs", {})
        return kwargs.get("source") or kwargs.get("channel") or "unknown"
    except Exception as e:
        logger.warning(f"Error extracting feedback source: {e}")
        return None


def extract_conversation_id(arguments: Dict[str, Any]) -> Optional[str]:
    """
    Extract the conversation/session ID this feedback relates to.
    
    Args:
        arguments: Dict containing instance, args, kwargs, result
        
    Returns:
        Conversation or session ID
    """
    try:
        kwargs = arguments.get("kwargs", {})
        return (kwargs.get("conversation_id") or 
                kwargs.get("session_id") or 
                kwargs.get("thread_id"))
    except Exception as e:
        logger.warning(f"Error extracting conversation ID: {e}")
        return None


def extract_message_id(arguments: Dict[str, Any]) -> Optional[str]:
    """
    Extract the specific message/turn ID being evaluated.
    
    Args:
        arguments: Dict containing instance, args, kwargs, result
        
    Returns:
        Message or turn ID
    """
    try:
        kwargs = arguments.get("kwargs", {})
        return (kwargs.get("message_id") or 
                kwargs.get("turn_id") or 
                kwargs.get("response_id"))
    except Exception as e:
        logger.warning(f"Error extracting message ID: {e}")
        return None


def extract_user_id(arguments: Dict[str, Any]) -> Optional[str]:
    """
    Extract the user ID providing feedback.
    
    Args:
        arguments: Dict containing instance, args, kwargs, result
        
    Returns:
        User identifier
    """
    try:
        kwargs = arguments.get("kwargs", {})
        return kwargs.get("user_id") or kwargs.get("user") or "anonymous"
    except Exception as e:
        logger.warning(f"Error extracting user ID: {e}")
        return None


def extract_feedback_context(arguments: Dict[str, Any]) -> Optional[str]:
    """
    Extract the conversation context that prompted this feedback.
    
    Args:
        arguments: Dict containing instance, args, kwargs, result
        
    Returns:
        JSON string of conversation context
    """
    try:
        kwargs = arguments.get("kwargs", {})
        context = kwargs.get("context") or kwargs.get("conversation_history")
        
        if context:
            if isinstance(context, str):
                return context
            return get_json_dumps(context)
        return None
    except Exception as e:
        logger.warning(f"Error extracting feedback context: {e}")
        return None


def extract_evaluated_response(arguments: Dict[str, Any]) -> Optional[str]:
    """
    Extract the agent response being evaluated by the user.
    
    Args:
        arguments: Dict containing instance, args, kwargs, result
        
    Returns:
        The agent's response that is being rated
    """
    try:
        kwargs = arguments.get("kwargs", {})
        response = (kwargs.get("evaluated_response") or 
                   kwargs.get("agent_response") or 
                   kwargs.get("bot_message"))
        
        if response:
            if isinstance(response, str):
                return response
            return get_json_dumps(response)
        return None
    except Exception as e:
        logger.warning(f"Error extracting evaluated response: {e}")
        return None


def extract_feedback_rating(arguments: Dict[str, Any]) -> Optional[str]:
    """
    Extract the user's satisfaction rating.
    
    Args:
        arguments: Dict containing instance, args, kwargs, result
        
    Returns:
        Rating value (e.g., 'satisfied', '5', 'thumbs_up')
    """
    try:
        status = get_status_code(arguments)
        if status != 'success':
            return None
            
        result = arguments.get("result")
        
        # Try multiple field names for rating
        if hasattr(result, "rating"):
            return str(result.rating)
        elif isinstance(result, dict):
            rating = (result.get("rating") or 
                     result.get("satisfaction") or 
                     result.get("score"))
            return str(rating) if rating is not None else None
        
        # Check kwargs as fallback
        kwargs = arguments.get("kwargs", {})
        rating = (kwargs.get("rating") or 
                 kwargs.get("satisfaction") or 
                 kwargs.get("score"))
        return str(rating) if rating is not None else None
        
    except Exception as e:
        logger.warning(f"Error extracting feedback rating: {e}")
        return None


def extract_rating_type(arguments: Dict[str, Any]) -> Optional[str]:
    """
    Extract the type of rating system used.
    
    Args:
        arguments: Dict containing instance, args, kwargs, result
        
    Returns:
        Rating type (e.g., 'binary', 'scale_1_5', 'thumbs', 'stars')
    """
    try:
        kwargs = arguments.get("kwargs", {})
        rating_type = kwargs.get("rating_type")
        
        if rating_type:
            return rating_type
            
        # Try to infer from rating value
        rating = extract_feedback_rating(arguments)
        if rating:
            rating_lower = rating.lower()
            if rating_lower in ['satisfied', 'unsatisfied', 'yes', 'no', 'thumbs_up', 'thumbs_down']:
                return 'binary'
            elif rating_lower.replace('.', '').isdigit():
                float_val = float(rating)
                if 1 <= float_val <= 5:
                    return 'scale_1_5'
                elif 1 <= float_val <= 10:
                    return 'scale_1_10'
                return 'numeric'
        
        return 'unknown'
    except Exception as e:
        logger.warning(f"Error extracting rating type: {e}")
        return None


def extract_feedback_comment(arguments: Dict[str, Any]) -> Optional[str]:
    """
    Extract the user's feedback comment or explanation.
    
    Args:
        arguments: Dict containing instance, args, kwargs, result
        
    Returns:
        User's comment text
    """
    try:
        status = get_status_code(arguments)
        if status != 'success':
            return None
            
        result = arguments.get("result")
        
        if hasattr(result, "comment"):
            return result.comment
        elif hasattr(result, "feedback"):
            return result.feedback
        elif isinstance(result, dict):
            return (result.get("comment") or 
                   result.get("feedback") or 
                   result.get("text") or
                   result.get("message"))
        
        # Check kwargs as fallback
        kwargs = arguments.get("kwargs", {})
        return (kwargs.get("comment") or 
               kwargs.get("feedback") or 
               kwargs.get("text"))
        
    except Exception as e:
        logger.warning(f"Error extracting feedback comment: {e}")
        return None


def extract_sentiment(arguments: Dict[str, Any]) -> Optional[str]:
    """
    Extract or derive sentiment from the feedback.
    
    Args:
        arguments: Dict containing instance, args, kwargs, result
        
    Returns:
        Sentiment classification ('positive', 'negative', 'neutral')
    """
    try:
        result = arguments.get("result")
        
        # First try explicit sentiment field
        if hasattr(result, "sentiment"):
            return result.sentiment
        elif isinstance(result, dict) and result.get("sentiment"):
            return result.get("sentiment")
        
        # Derive from rating
        rating = extract_feedback_rating(arguments)
        if rating:
            rating_lower = rating.lower()
            
            # Binary sentiment
            if rating_lower in ['satisfied', 'yes', 'thumbs_up', 'positive']:
                return 'positive'
            elif rating_lower in ['unsatisfied', 'no', 'thumbs_down', 'negative']:
                return 'negative'
            
            # Numeric sentiment
            if rating_lower.replace('.', '').isdigit():
                float_val = float(rating)
                if float_val >= 4:  # Assuming 5-point scale
                    return 'positive'
                elif float_val <= 2:
                    return 'negative'
                else:
                    return 'neutral'
        
        return 'neutral'
    except Exception as e:
        logger.warning(f"Error extracting sentiment: {e}")
        return None


def extract_feedback_metadata(arguments: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Extract additional metadata about the feedback collection.
    
    Args:
        arguments: Dict containing instance, args, kwargs, result
        
    Returns:
        Dict of metadata (timestamp, collection_method, etc.)
    """
    try:
        metadata = {}
        
        result = arguments.get("result")
        kwargs = arguments.get("kwargs", {})
        
        # Timestamp
        if hasattr(result, "timestamp"):
            metadata["timestamp"] = result.timestamp
        elif isinstance(result, dict) and result.get("timestamp"):
            metadata["timestamp"] = result.get("timestamp")
        elif kwargs.get("timestamp"):
            metadata["timestamp"] = kwargs.get("timestamp")
        
        # Collection method
        collection_method = kwargs.get("collection_method") or kwargs.get("method")
        if collection_method:
            metadata["collection_method"] = collection_method
        
        # Tags or categories
        tags = kwargs.get("tags") or kwargs.get("categories")
        if tags:
            metadata["tags"] = tags if isinstance(tags, str) else get_json_dumps(tags)
        
        # Additional custom fields
        custom_fields = kwargs.get("metadata") or kwargs.get("custom_data")
        if custom_fields and isinstance(custom_fields, dict):
            metadata.update(custom_fields)
        
        return metadata if metadata else None
        
    except Exception as e:
        logger.warning(f"Error extracting feedback metadata: {e}")
        return None
