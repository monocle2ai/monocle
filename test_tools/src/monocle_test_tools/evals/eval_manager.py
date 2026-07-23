from typing import Optional, Union

from monocle_test_tools.evals.base_eval import BaseEval
from monocle_test_tools.evals.bert_eval import BertScorerEval
from monocle_test_tools.evals.exact_match_eval import ExactMatchEval
from monocle_test_tools.evals.json_validity_eval import JSONValidityEval
from monocle_test_tools.evals.keyword_presence_eval import KeywordPresenceEval
from monocle_test_tools.evals.okahu_eval import OkahuEval
from monocle_test_tools.evals.regex_match_eval import RegexMatchEval

# String keys for the built-in, non-LLM evaluators.
NON_LLM_EVALS = {
    "regex_match": RegexMatchEval,
    "json_validity": JSONValidityEval,
    "keyword_presence": KeywordPresenceEval,
    "exact_match": ExactMatchEval,
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
                raise ValueError(f"Invalid eval class name: {eval}. Error: {e}") from e
    return eval


# Evaluators selectable as an ``eval_source`` (name -> BaseEval subclass). Add new
# eval providers here; each may customize behavior via the BaseEval interface
EVAL_SOURCE_CLASSES: dict = {
    "okahu": OkahuEval,
}


def get_supported_eval_sources() -> tuple:
    """Names accepted as an ``eval_source``."""
    return tuple(EVAL_SOURCE_CLASSES.keys())


def get_evaluator_class(eval_source: str):
    """Return the ``BaseEval`` subclass for an eval source name (e.g. ``"okahu"``).

    Raises ``ValueError`` for an unknown source.
    """
    try:
        return EVAL_SOURCE_CLASSES[eval_source]
    except KeyError:
        raise ValueError(
            f"Unsupported eval_source: '{eval_source}'. "
            f"Supported values: {', '.join(sorted(EVAL_SOURCE_CLASSES))}."
        ) from None
