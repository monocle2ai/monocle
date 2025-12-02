
from typing import Optional, Union
from monocle_test_tools.comparer.base_comparer import BaseComparer

def get_comparer(comparer: Optional[Union[str, BaseComparer]]) -> BaseComparer:
    if isinstance(comparer, str):
        if comparer == "default":
            from monocle_test_tools.comparer.default_comparer import DefaultComparer
            comparer = DefaultComparer()
        elif comparer == "similarity":
            from monocle_test_tools.comparer.sentense_comparer import SentenceComparer
            comparer = SentenceComparer()
        elif comparer == "bert_score":
            from monocle_test_tools.comparer.bert_score_comparer import BertScoreComparer
            comparer = BertScoreComparer()
        elif comparer == "metric":
            from monocle_test_tools.comparer.metric_comparer import MetricComparer
            comparer = MetricComparer()
        elif comparer == "token_match":
            from monocle_test_tools.comparer.token_match_comparer import TokenMatch
            comparer = TokenMatch()
        else:
            try:
                # instantiate comparer class from string
                comparer_class = globals()[comparer]
                if issubclass(comparer_class, BaseComparer):
                    comparer = comparer_class()
            except Exception as e:
                raise ValueError(f"Invalid comparer class name: {comparer}. Error: {e}")
    elif comparer is None:
        comparer = DefaultComparer()
    return comparer
