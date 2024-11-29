"""
This module provides utility functions for extracting system, user,
and assistant messages from various input formats.
"""

import logging
from monocle_apptrace.utils import get_attribute
DATA_INPUT_KEY = "data.input"

logger = logging.getLogger(__name__)
def extract_messages(args):
    """Extract system and user messages"""
    try:
        system_message, user_message = "", ""
        args_input = get_attribute(DATA_INPUT_KEY)
        if args_input:
            user_message = args_input
            return system_message, user_message
        if args and isinstance(args, tuple) and len(args) > 0:
            if hasattr(args[0], "messages") and isinstance(args[0].messages, list):
                for msg in args[0].messages:
                    if hasattr(msg, 'content') and hasattr(msg, 'type'):
                        if msg.type == "system":
                            system_message = msg.content
                        elif msg.type in ["user", "human"]:
                            user_message = msg.content
            elif isinstance(args[0], list):
                for msg in args[0]:
                    if hasattr(msg, 'content') and hasattr(msg, 'role'):
                        if msg.role == "system":
                            system_message = msg.content
                        elif msg.role in ["user", "human"]:
                            user_message = extract_query_from_content(msg.content)
        return system_message, user_message
    except Exception as e:
        logger.warning("Warning: Error occurred in extract_messages: %s", str(e))
        return "", ""


def extract_assistant_message(response):
    try:
        if isinstance(response, str):
            return response
        if hasattr(response, "content"):
            return response.content
        if hasattr(response, "message") and hasattr(response.message, "content"):
            return response.message.content
        if "replies" in response:
            if hasattr(response['replies'][0], 'content'):
                return response['replies'][0].content
            else:
                return response['replies'][0]
        return ""
    except Exception as e:
        logger.warning("Warning: Error occurred in extract_assistant_message: %s", str(e))
        return ""


def extract_query_from_content(content):
    try:
        query_prefix = "Query:"
        answer_prefix = "Answer:"
        query_start = content.find(query_prefix)
        if query_start == -1:
            return None

        query_start += len(query_prefix)
        answer_start = content.find(answer_prefix, query_start)
        if answer_start == -1:
            query = content[query_start:].strip()
        else:
            query = content[query_start:answer_start].strip()
        return query
    except Exception as e:
        logger.warning("Warning: Error occurred in extract_query_from_content: %s", str(e))
        return ""
