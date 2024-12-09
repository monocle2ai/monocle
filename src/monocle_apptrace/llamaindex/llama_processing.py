"""
This module provides utility functions for extracting system, user,
and assistant messages for llama index openai.
"""
import logging
logger = logging.getLogger(__name__)


def llama_processing(arguments):
    accessor = arguments["accessor"]
    if "arguments" in accessor:
        return extract_messages(arguments['args'])
    if "response" in accessor:
        return extract_assistant_message(arguments['output'])


def extract_messages(args):
    try:
        messages = []
        if isinstance(args[0], list):
            for msg in args[0]:
                if hasattr(msg, 'content') and hasattr(msg, 'role'):
                    if hasattr(msg.role, 'value'):
                        role = msg.role.value
                    else:
                        role = msg.role
                    if msg.role == "system":
                        messages.append({role: msg.content})
                    elif msg.role in ["user", "human"]:
                        user_message = extract_query_from_content(msg.content)
                        messages.append({role: user_message})
        return messages
    except Exception as e:
        logger.warning("Warning: Error occurred in extract_messages: %s", str(e))
        return []


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


def extract_assistant_message(response):
    try:
        if hasattr(response, "message") and hasattr(response.message, "content"):
            if hasattr(response.message, "role"):
                if hasattr(response.message.role, 'value'):
                    role = response.message.role.value
                else:
                    role = response.message.role
                return [{role: response.message.content}]
            return [response.message.content]
        if isinstance(response, dict):
            return [response]
        return []
    except Exception as e:
        logger.warning("Warning: Error occurred in extract_assistant_message: %s", str(e))
        return []

