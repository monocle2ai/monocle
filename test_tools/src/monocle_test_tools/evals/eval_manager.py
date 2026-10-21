from typing import Optional, Type, Union

from monocle_apptrace.providers import NamedRegistry
from monocle_test_tools.evals.base_eval import BaseEval
from monocle_test_tools.evals.bert_eval import BertScorerEval
from monocle_test_tools.evals.exact_match_eval import ExactMatchEval
from monocle_test_tools.evals.json_validity_eval import JSONValidityEval
from monocle_test_tools.evals.keyword_presence_eval import KeywordPresenceEval
from monocle_test_tools.evals.regex_match_eval import RegexMatchEval

# String keys for the built-in, non-LLM evaluators.
NON_LLM_EVALS = {
    "regex_match": RegexMatchEval,
    "json_validity": JSONValidityEval,
    "keyword_presence": KeywordPresenceEval,
    "exact_match": ExactMatchEval,
}

# Evaluators that come from providers. Each is also an ``eval_source``.
_eval_sources = NamedRegistry("monocle.evals", {
    "okahu": "monocle_test_tools.providers.okahu.okahu_eval:OkahuEval",
})

def get_evaluator(eval: Optional[Union[str, BaseEval]], eval_options: Optional[dict] = None) -> BaseEval:
    if isinstance(eval, str):
        if eval == "bert_score":
            eval = BertScorerEval(eval_options=eval_options)
        elif eval in NON_LLM_EVALS:
            eval = NON_LLM_EVALS[eval](eval_options=eval_options)
        elif get_eval_source_class(eval) is not None:
            eval = get_eval_source_class(eval)(eval_options=eval_options)
        else:
            try:
                eval_class = globals()[eval]
                if issubclass(eval_class, BaseEval):
                    eval = eval_class(eval_options=eval_options)
            except Exception as e:
                raise ValueError(f"Invalid eval class name: {eval}. Error: {e}") from e
    return eval


def get_eval_source_class(name: str) -> Optional[Type[BaseEval]]:
    """The evaluator class for an ``eval_source`` name, or None if there is none."""
    return _eval_sources.get(name)


def get_supported_eval_sources() -> tuple:
    """Names accepted as an ``eval_source``."""
    return tuple(_eval_sources.names())
