from monocle_test_tools.comparer.base_comparer import BaseComparer

class TokenMatchComparer(BaseComparer):
    def compare(self, expected: str, actual: str) -> bool:
        if expected == actual:
            return True
        if expected is None or actual is None:
            return False
        return expected in actual