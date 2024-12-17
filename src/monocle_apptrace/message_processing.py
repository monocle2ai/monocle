"""
This module provides utility functions for extracting system, user,
and assistant messages from various input formats.
"""

import logging
import json
from io import BytesIO
from monocle_apptrace.utils import get_attribute
DATA_INPUT_KEY = "data.input"

logger = logging.getLogger(__name__)
def extract_messages(args):
    """Extract system and user messages"""
    try:
        messages = []
        args_input = get_attribute(DATA_INPUT_KEY)
        if args_input:
            messages.append(args_input)
            return messages
        if args and isinstance(args, tuple) and len(args) > 0:
            if hasattr(args[0], "messages") and isinstance(args[0].messages, list):
                for msg in args[0].messages:
                    if hasattr(msg, 'content') and hasattr(msg, 'type'):
                        messages.append({msg.type: msg.content})
            elif isinstance(args[0], list):  #llama
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
        elif args and isinstance(args, dict) and len(args) > 0:
            if 'Body' in args and isinstance(args['Body'], str):
                data = json.loads(args['Body'])
                question = data.get("question")
                messages.append(question)

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
        if hasattr(response, "message") and hasattr(response.message, "content"):
            return [response.message.content]
        if "replies" in response:
            reply = response["replies"][0]
            if hasattr(reply, 'content'):
                return [reply.content]
            return [reply]
        if "Body" in response and hasattr(response['Body'], "_raw_stream"):
            raw_stream = getattr(response['Body'], "_raw_stream")
            if hasattr(raw_stream, "data"):
                response_bytes = getattr(raw_stream, "data")
                response_str = response_bytes.decode('utf-8')
                response_dict = json.loads(response_str)
                response['Body'] = BytesIO(response_bytes)
                return [response_dict]
        if isinstance(response, dict):
            return [response]
        return []
    except Exception as e:
        logger.warning("Warning: Error occurred in extract_assistant_message: %s", str(e))
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
