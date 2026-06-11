from typing import Optional, Union
from monocle_test_tools.evals.base_eval import BaseEval
from monocle_test_tools.evals.bert_eval import BertScorerEval
from monocle_test_tools.evals.okahu_eval import OkahuEval
from monocle_test_tools.evals.regex_match_eval import RegexMatchEval
from monocle_test_tools.evals.json_validity_eval import JSONValidityEval
from monocle_test_tools.evals.keyword_presence_eval import KeywordPresenceEval
from monocle_test_tools.evals.exact_match_eval import ExactMatchEval
from monocle_test_tools.evals.pii_detection_eval import PIIDetectionEval
from monocle_test_tools.evals.readability_eval import ReadabilityEval
from monocle_test_tools.evals.token_overlap_eval import TokenOverlapEval

# String keys for the built-in, non-LLM evaluators.
NON_LLM_EVALS = {
    "regex_match": RegexMatchEval,
    "json_validity": JSONValidityEval,
    "keyword_presence": KeywordPresenceEval,
    "exact_match": ExactMatchEval,
    "pii_detection": PIIDetectionEval,
    "readability": ReadabilityEval,
    "token_overlap": TokenOverlapEval,
}

def get_evaluator(eval: Optional[Union[str, BaseEval]], eval_options: Optional[dict] = None) -> BaseEval:
    if isinstance(eval, str):
        if eval == "okahu":
            eval = OkahuEval(eval_options=eval_options)
        elif eval == "bert_score":
            eval = BertScorerEval(eval_options=eval_options)
        elif eval in NON_LLM_EVALS:
            eval = NON_LLM_EVALS[eval](eval_options=eval_options)
        else:
            try:
                eval_class = globals()[eval]
                if issubclass(eval_class, BaseEval):
                    eval = eval_class(eval_options=eval_options)
            except Exception as e:
                raise ValueError(f"Invalid eval class name: {eval}. Error: {e}")
    return eval
