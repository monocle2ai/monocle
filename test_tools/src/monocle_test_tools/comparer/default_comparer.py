from monocle_test_tools.comparer.base_comparer import BaseComparer

class DefaultComparer(BaseComparer):
    def compare(self, expected: str, actual: str) -> bool:
        return expected.strip() == actual.strip()