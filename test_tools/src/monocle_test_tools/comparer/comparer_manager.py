
from typing import Optional, Union
from monocle_test_tools.comparer.base_comparer import BaseComparer
from monocle_test_tools.comparer.default_comparer import DefaultComparer
from monocle_test_tools.comparer.bert_score_comparer import BertScoreComparer
from monocle_test_tools.comparer.metric_comparer import MetricComparer

def get_comparer(comparer: Optional[Union[str, BaseComparer]]) -> BaseComparer:
    if isinstance(comparer, str):
        if comparer == "default":
            comparer = DefaultComparer()
        elif comparer == "similarity":
            comparer = BertScoreComparer()
        elif comparer == "metric":
            comparer = MetricComparer()
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
