"""
This module provides utility functions for extracting system, user,
and assistant messages for haystack  openai.
"""
import logging
logger = logging.getLogger(__name__)
DATA_INPUT_KEY = "data.input"
from monocle_apptrace.utils import get_attribute


def haystack_processing(arguments):
    accessor = arguments["accessor"]
    if "arguments" in accessor:
        return extract_messages(arguments['args'])
    if "response" in accessor:
        return extract_assistant_message(arguments['output'])


def extract_messages(args):
    try:
        messages = []
        args_input = get_attribute(DATA_INPUT_KEY)
        if args_input:
            messages.append(args_input)
            return messages
    except Exception as e:
        logger.warning("Warning: Error occurred in extract_messages: %s", str(e))
        return []


def extract_assistant_message(response):
    try:
        if "replies" in response:
            reply = response["replies"][0]
            if hasattr(reply, 'content'):
                return [reply.content]
            return [reply]
    except Exception as e:
        logger.warning("Warning: Error occurred in extract_assistant_message: %s", str(e))
        return []
