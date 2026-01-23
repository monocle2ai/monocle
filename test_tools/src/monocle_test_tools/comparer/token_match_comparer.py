from monocle_test_tools.comparer.base_comparer import BaseComparer

class TokenMatchComparer(BaseComparer):
    def compare(self, expected: str, actual: str) -> bool:
        if expected == actual:
            return True
        if expected is None or actual is None:
            return False
        # Case-insensitive substring match
        # Convert to string if other types
        expected_str = str(expected) if not isinstance(expected, str) else expected
        actual_str = str(actual) if not isinstance(actual, str) else actual
        return expected_str.lower() in actual_str.lower()