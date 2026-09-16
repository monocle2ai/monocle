"""Unit tests for the MS Agent Framework turn-input helper.

The turn span is the only one handed a raw ``Message``, so it is the only place
where rendering it wrongly shows up in a trace. It used to read ``content``
(singular), which agent-framework 1.6.0 renamed to ``contents`` — the guard
missed, ``str()`` ran on a class with no ``__str__``, and the span recorded
``<agent_framework._types.Message object at 0x...>`` instead of the prompt.

These tests use stand-in message objects rather than importing
agent_framework, so they run without it installed.
"""
import pytest

from monocle_apptrace.instrumentation.metamodel.msagent._helper import (
    extract_request_agent_input,
)


def render(value):
    """Render one positional turn input, the way the turn span does."""
    return extract_request_agent_input({"args": (value,), "kwargs": {}})

PROMPT = "Book a flight from Delhi to Phuket"


class MessageLike:
    """Stands in for agent-framework >= 1.6.0's Message.

    ``contents`` plural, a computed ``text``, and deliberately no ``__str__``,
    so ``str()`` on it produces the default object repr — the bug's signature.
    """

    def __init__(self, text=PROMPT, contents=None):
        self.contents = contents if contents is not None else [text]
        self._text = text

    @property
    def text(self):
        return self._text


class LegacyMessage:
    """Stands in for an older message object exposing ``content`` singular."""

    def __init__(self, content=PROMPT):
        self.content = content


class ContentLike:
    """Stands in for agent-framework's Content: to_dict(), and no __str__."""

    def __init__(self, **fields):
        self._fields = fields

    def to_dict(self):
        return dict(self._fields)


class ContentsOnlyMessage:
    """A message whose text cannot be computed — e.g. a tool-call-only turn."""

    def __init__(self, contents=None):
        self.contents = contents if contents is not None else [
            ContentLike(type="function_call", name="book_flight")
        ]
        self.text = ""


class Opaque:
    """No text, no contents, no __str__ — the genuine last resort."""


def assert_not_a_repr(value: str) -> None:
    assert "object at 0x" not in value, f"rendered as an object repr: {value}"


# -- rendering one turn input ----------------------------------------------

def test_string_is_returned_unchanged():
    assert render(PROMPT) == PROMPT


def test_message_with_contents_renders_its_text():
    rendered = render(MessageLike())
    assert rendered == PROMPT
    assert_not_a_repr(rendered)


def test_legacy_content_singular_is_still_supported():
    assert render(LegacyMessage()) == PROMPT


def test_message_without_usable_text_is_not_a_repr():
    """A turn carrying no text must still not be exported as an object repr."""
    assert_not_a_repr(render(ContentsOnlyMessage()))


def test_contents_whose_to_dict_raises_still_render():
    class Hostile:
        def to_dict(self):
            raise RuntimeError("boom")

    class Message:
        text = ""
        contents = [Hostile()]

    # Nothing better than the repr exists here; the point is that it does not raise.
    assert isinstance(render(Message()), str)


def test_non_string_text_attribute_is_not_trusted():
    """A `text` that is not a string must not be returned as the input.

    The contents are used instead, joined by the module's own
    ``_extract_text_from_content``.
    """
    class WeirdText:
        text = 42
        content = "fallback"

    assert render(WeirdText()) == "fallback"


def test_opaque_object_is_stringified_as_a_last_resort():
    # Nothing better exists here; the point is that it does not raise.
    assert isinstance(render(Opaque()), str)


# -- the accessor the span uses --------------------------------------------

def test_positional_message_is_rendered():
    rendered = extract_request_agent_input({"args": (MessageLike(),), "kwargs": {}})
    assert rendered == PROMPT
    assert_not_a_repr(rendered)


def test_positional_string_is_rendered():
    assert extract_request_agent_input({"args": (PROMPT,), "kwargs": {}}) == PROMPT


@pytest.mark.parametrize("key", ["input", "message", "query"])
def test_message_passed_by_keyword_is_rendered(key):
    rendered = extract_request_agent_input({"args": (), "kwargs": {key: MessageLike()}})
    assert rendered == PROMPT
    assert_not_a_repr(rendered)


def test_messages_list_renders_the_first_message():
    rendered = extract_request_agent_input(
        {"args": (), "kwargs": {"messages": [MessageLike(), MessageLike("second")]}}
    )
    assert rendered == PROMPT
    assert_not_a_repr(rendered)


def test_empty_messages_list_does_not_raise():
    assert isinstance(
        extract_request_agent_input({"args": (), "kwargs": {"messages": []}}), str
    )


def test_no_input_yields_empty_string():
    assert extract_request_agent_input({"args": (), "kwargs": {}}) == ""


def test_extraction_failure_is_swallowed():
    """The accessor must never break span creation."""
    class Exploding:
        @property
        def text(self):
            raise RuntimeError("boom")

    assert extract_request_agent_input({"args": (Exploding(),), "kwargs": {}}) == ""


if __name__ == "__main__":
    pytest.main([__file__])
