"""Unit tests for the decision-model finish_reason mapping."""
import unittest

from monocle_apptrace.instrumentation.metamodel.finish_types import (
    DECISION_FINISH_REASON_MAPPING,
    map_decision_finish_reason_to_finish_type,
)


class TestDecisionFinishReasonMapping(unittest.TestCase):
    def test_mapping_contents(self):
        self.assertEqual(DECISION_FINISH_REASON_MAPPING["success"], "success")
        self.assertEqual(DECISION_FINISH_REASON_MAPPING["stop"], "success")
        self.assertEqual(DECISION_FINISH_REASON_MAPPING["error"], "error")

    def test_map_success_and_error(self):
        self.assertEqual(map_decision_finish_reason_to_finish_type("success"), "success")
        self.assertEqual(map_decision_finish_reason_to_finish_type("stop"), "success")
        self.assertEqual(map_decision_finish_reason_to_finish_type("error"), "error")

    def test_map_none_and_unknown(self):
        self.assertIsNone(map_decision_finish_reason_to_finish_type(None))
        self.assertIsNone(map_decision_finish_reason_to_finish_type(""))
        self.assertIsNone(map_decision_finish_reason_to_finish_type("weird"))


if __name__ == "__main__":
    unittest.main()
