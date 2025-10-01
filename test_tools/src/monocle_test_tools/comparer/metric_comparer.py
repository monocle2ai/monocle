import logging
from monocle_test_tools.comparer.base_comparer import BaseComparer
from typing import Optional, Union

class MetricComparer(BaseComparer):
    def compare(self, expected: Union[dict, str], actual: Union[dict, str]) -> bool:
        if not isinstance(expected, dict) or not isinstance(actual, dict):
            raise ValueError("Expected and actual values must be dictionaries.")
        for key, expected_value in expected.items():
            actual_value = actual.get(key)
            if expected_value > actual_value:
                logging.debug(f"Mismatch for key '{key}': expected '{expected_value}', got '{actual_value}'")
                return False
        return True