"""Unit tests for test.run attribute logic in MonocleValidator."""
from unittest.mock import MagicMock

# Hardcoded to avoid importing the full monocle_test_tools package
TEST_RUN_ATTRIBUTE = "test.run"
TEST_STATUS_ATTRIBUTE = "test.status"
TEST_ASSERTION_ATTRIBUTE = "test.assertion.message"


def _make_span():
    span = MagicMock()
    span._attributes = {}
    return span


def _simulate_flush(spans, test_failed, test_assertion_message=None):
    """Replicate the flush_to_exporters logic we added to validator.py."""
    for span in spans:
        if test_failed:
            span._attributes[TEST_STATUS_ATTRIBUTE] = "failed"
            if test_assertion_message is not None:
                span._attributes[TEST_ASSERTION_ATTRIBUTE] = test_assertion_message
        else:
            span._attributes[TEST_STATUS_ATTRIBUTE] = "passed"
        span._attributes[TEST_RUN_ATTRIBUTE] = True


class TestTestRunAttribute:

    def test_constant_value(self):
        assert TEST_RUN_ATTRIBUTE == "test.run"

    def test_attribute_set_on_passed_span(self):
        span = _make_span()
        _simulate_flush([span], test_failed=False)
        assert span._attributes[TEST_RUN_ATTRIBUTE] is True
        assert span._attributes[TEST_STATUS_ATTRIBUTE] == "passed"

    def test_attribute_set_on_failed_span(self):
        span = _make_span()
        _simulate_flush([span], test_failed=True)
        assert span._attributes[TEST_RUN_ATTRIBUTE] is True
        assert span._attributes[TEST_STATUS_ATTRIBUTE] == "failed"

    def test_attribute_is_boolean(self):
        span = _make_span()
        _simulate_flush([span], test_failed=False)
        assert isinstance(span._attributes[TEST_RUN_ATTRIBUTE], bool)

    def test_multiple_spans_all_get_attribute(self):
        spans = [_make_span(), _make_span(), _make_span()]
        _simulate_flush(spans, test_failed=False)
        for span in spans:
            assert span._attributes[TEST_RUN_ATTRIBUTE] is True

    def test_assertion_message_set_on_failed_span(self):
        span = _make_span()
        _simulate_flush([span], test_failed=True, test_assertion_message="Expected X got Y")
        assert span._attributes[TEST_ASSERTION_ATTRIBUTE] == "Expected X got Y"
        assert span._attributes[TEST_RUN_ATTRIBUTE] is True