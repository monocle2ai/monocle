"""
This module provides utility functions for extracting system, user,
and assistant messages for langchain  openai.
"""
import logging
logger = logging.getLogger(__name__)


def langchain_processing(arguments):
    accessor = arguments["accessor"]
    if "arguments" in accessor:
        return extract_messages(arguments['args'])
    if "response" in accessor:
        return extract_assistant_message(arguments['output'])


def extract_messages(args):
    try:
        messages = []
        if args and isinstance(args, tuple) and len(args) > 0:
            if hasattr(args[0], "messages") and isinstance(args[0].messages, list):
                for msg in args[0].messages:
                    if hasattr(msg, 'content') and hasattr(msg, 'type'):
                        messages.append({msg.type: msg.content})
        return messages
    except Exception as e:
        logger.warning("Warning: Error occurred in extract_messages: %s", str(e))
        return []


def extract_assistant_message(response):
    try:
        if isinstance(response, str):
            return [response]
        if hasattr(response, "content"):
            return [response.content]
        if isinstance(response, dict):
            return [response]
        return []
    except Exception as e:
        logger.warning("Warning: Error occurred in extract_assistant_message: %s", str(e))
        return []
