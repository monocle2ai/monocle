from typing import Optional, Union
from monocle_test_tools.evals.base_eval import BaseEval
from monocle_test_tools.evals.bert_eval import BertScorerEval
from monocle_test_tools.evals.okahu_eval import OkahuEval

def get_evaluator(eval: Optional[Union[str, BaseEval]], eval_options: Optional[dict] = None) -> BaseEval:
    if isinstance(eval, str):
        if eval == "okahu":
            eval = OkahuEval(eval_options=eval_options)
        elif eval == "bert_score":
            eval = BertScorerEval(eval_options=eval_options)
        else:
            try:
                eval_class = globals()[eval]
                if issubclass(eval_class, BaseEval):
                    eval = eval_class(eval_options=eval_options)
            except Exception as e:
                raise ValueError(f"Invalid eval class name: {eval}. Error: {e}")
    return eval
