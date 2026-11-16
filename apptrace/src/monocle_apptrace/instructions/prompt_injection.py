"""The check for prompt injection: which part of a model call is new input, and the reason to stop on it."""
from collections.abc import Mapping
from typing import Iterable, List, Optional

from monocle_apptrace.instructions.classifier import injection_score

ROLES_CHECKED = ("user", "tool", "function")
# The default of LLM Guard, the project the default classifier comes from.
INJECTION_THRESHOLD = 0.92


def prompt_injection_found_in(conversation: Iterable, classifier_model: str) -> Optional[str]:
    """The reason to stop a call before it is sent, or None: the classifier labels its new input a prompt injection."""
    score = max((injection_score(text, classifier_model) for text in new_input(conversation)), default=0.0)
    if score < INJECTION_THRESHOLD:
        return None
    return f"Prompt injection detected with score {score:.2f}."


def new_input(conversation: Iterable) -> List[str]:
    """What the roles in ROLES_CHECKED wrote since the model last replied, one text for each piece.

    The conversation is in the chat shape: messages that hold a ``role`` and a ``content``, where the content is
    text or a list of parts that hold ``text``. The system prompt and the model's own messages are never new input.
    """
    checked = [message for message in messages_since_the_model_replied(conversation) if message.get("role") in ROLES_CHECKED]
    return [text for message in checked for text in texts_of(message.get("content")) if text.strip()]


def messages_since_the_model_replied(conversation: Iterable) -> List[Mapping]:
    """The messages that come after the model's last reply, newest first.

    Assistant messages that end the conversation are not a reply: the app wrote them for the model to carry on from.
    """
    unanswered = []
    for message in reversed(list(conversation)):
        fields = fields_of(message)
        from_the_model = fields.get("role") == "assistant"
        if from_the_model and unanswered:
            break
        if not from_the_model:
            unanswered.append(fields)
    return unanswered


def texts_of(content) -> List[str]:
    """The text of a message's content: the content itself, or the ``text`` of each part it lists."""
    if isinstance(content, str):
        return [content]
    if not isinstance(content, (list, tuple)):
        return []
    texts = [fields_of(part).get("text") for part in content]
    return [text for text in texts if isinstance(text, str)]


def fields_of(item) -> Mapping:
    """A message, or a part of one, as a mapping, whether the app wrote it as a dict or as an SDK object."""
    if isinstance(item, Mapping):
        return item
    if hasattr(item, "model_dump") and not isinstance(item, type):
        return item.model_dump()
    return {}


def chat_messages(kwargs: dict) -> list:
    """For a chat call that takes ``messages`` in the chat shape: its conversation."""
    return list(kwargs.get("messages") or [])
