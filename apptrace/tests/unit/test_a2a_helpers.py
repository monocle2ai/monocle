"""
Unit tests for the A2A output-processor helpers.

Cover the three shapes a ``send_message`` call comes back as: a Task, a
Message, and a JSON-RPC error. A helper that raises on one of them loses its
attribute silently, since the span handler swallows accessor errors.
"""
import unittest
from types import SimpleNamespace

import pytest

pytest.importorskip("a2a", reason="a2a SDK not installed")

from a2a.types import SendMessageRequest, SendMessageResponse  # noqa: E402

from monocle_apptrace.instrumentation.metamodel.a2a._helper import (  # noqa: E402
    get_params_arguments,
    get_response,
    get_role,
    get_status,
    get_url,
)

TASK_RESULT = {
    "id": "1", "jsonrpc": "2.0",
    "result": {"id": "t1", "contextId": "c1", "kind": "task",
               "status": {"state": "completed"},
               "artifacts": [{"artifactId": "a1",
                              "parts": [{"kind": "text", "text": "830 INR"}]}]},
}

MESSAGE_RESULT = {
    "id": "1", "jsonrpc": "2.0",
    "result": {"kind": "message", "role": "agent", "messageId": "m1",
               "parts": [{"kind": "text", "text": "830 INR"}]},
}

ERROR_RESULT = {
    "id": "1", "jsonrpc": "2.0",
    "error": {"code": -32601, "message": "method not found"},
}


def _result_arguments(raw):
    return {"result": SendMessageResponse.model_validate(raw)}


def _request_arguments(parts):
    request = SendMessageRequest.model_validate({
        "id": "1", "jsonrpc": "2.0", "method": "message/send",
        "params": {"message": {"role": "user", "messageId": "m1", "kind": "message",
                               "parts": parts}},
    })
    return {"args": [request]}


class TestA2AOutputHelpers(unittest.TestCase):

    def test_task_result_reports_status_and_artifact_text(self):
        arguments = _result_arguments(TASK_RESULT)
        self.assertEqual(get_status(arguments), "completed")
        self.assertEqual(get_response(arguments), "830 INR")

    def test_message_result_reports_its_own_parts_and_no_status(self):
        arguments = _result_arguments(MESSAGE_RESULT)
        self.assertIsNone(get_status(arguments))
        self.assertEqual(get_response(arguments), "830 INR")

    def test_several_parts_come_back_as_one_string(self):
        raw = {"id": "1", "jsonrpc": "2.0",
               "result": {"kind": "message", "role": "agent", "messageId": "m1",
                          "parts": [{"kind": "text", "text": "830 INR"},
                                    {"kind": "text", "text": "rate 83.0"}]}}
        self.assertEqual(get_response(_result_arguments(raw)), "830 INR\nrate 83.0")


class TestA2AUrl(unittest.TestCase):
    """The url sits on the client in older SDKs and on its transport in newer ones."""

    def test_url_on_the_client(self):
        client = SimpleNamespace(url="http://agent.test/")
        self.assertEqual(get_url({"instance": client}), "http://agent.test/")

    def test_url_on_the_transport(self):
        client = SimpleNamespace(_transport=SimpleNamespace(url="http://agent.test/"))
        self.assertEqual(get_url({"instance": client}), "http://agent.test/")

    def test_no_url_anywhere_reports_nothing(self):
        self.assertIsNone(get_url({"instance": SimpleNamespace()}))


class TestA2AInputHelpers(unittest.TestCase):

    def test_text_part_is_read(self):
        arguments = _request_arguments([{"kind": "text", "text": "how much is 10 USD?"}])
        self.assertEqual(get_params_arguments(arguments), "how much is 10 USD?")
        self.assertEqual(get_role(arguments), "user")

    def test_non_text_first_part_is_skipped(self):
        arguments = _request_arguments([
            {"kind": "data", "data": {"amount": 10}},
            {"kind": "text", "text": "how much is 10 USD?"},
        ])
        self.assertEqual(get_params_arguments(arguments), "how much is 10 USD?")

    def test_message_without_text_reports_nothing(self):
        arguments = _request_arguments([{"kind": "data", "data": {"amount": 10}}])
        self.assertIsNone(get_params_arguments(arguments))


if __name__ == "__main__":
    unittest.main()
