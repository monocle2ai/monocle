import logging
import math
import re
from collections import Counter
from monocle_test_tools.evals.base_eval import BaseEval
from pydantic import Field

logger = logging.getLogger(__name__)

# Tiny constant used to smooth zero-count higher-order n-gram precisions so a
# single missing n-gram order does not collapse the whole score to 0.
_EPSILON = 1e-9


def _ngram_counts(tokens: list, n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


class BleuEval(BaseEval):
    """
    Non-LLM evaluator that computes sentence-level BLEU between a reference
    (``input``) and a candidate (``output``). BLEU is the standard precision-based
    n-gram overlap metric used for translation/generation quality. This is a pure,
    deterministic implementation — no model, no network, no external dependency.

    eval_options:
        max_n (int): Maximum n-gram order. Default 4 (i.e. BLEU-4).
        weights (list[float]): Per-order weights; defaults to uniform 1/max_n.
        smooth (bool): Apply epsilon smoothing to zero-count orders. Default True.
        ignore_case (bool): Lowercase both sides before tokenizing. Default True.
        token_pattern (str): Regex used to extract tokens. Default word characters.

    Returns:
        bleu (float): BLEU score in [0.0, 1.0].
        brevity_penalty (float): The brevity penalty applied.
        precision_<n> (float): Modified precision for each n-gram order.
    """
    max_n: int = Field(default=4, description="Maximum n-gram order (BLEU-N).")
    weights: list = Field(default=None, description="Per-order weights; defaults to uniform.")
    smooth: bool = Field(default=True, description="Apply epsilon smoothing to zero-count orders.")
    ignore_case: bool = Field(default=True, description="Lowercase both sides before tokenizing.")
    token_pattern: str = Field(default=r"[A-Za-z0-9']+", description="Regex used to extract tokens.")

    def __init__(self, **data):
        super().__init__(**data)
        self.max_n = self.eval_options.get("max_n", self.max_n)
        self.weights = self.eval_options.get("weights", self.weights)
        self.smooth = self.eval_options.get("smooth", self.smooth)
        self.ignore_case = self.eval_options.get("ignore_case", self.ignore_case)
        self.token_pattern = self.eval_options.get("token_pattern", self.token_pattern)

    def _tokenize(self, value: str) -> list:
        text = value if isinstance(value, str) else str(value)
        if self.ignore_case:
            text = text.lower()
        return re.findall(self.token_pattern, text)

    def evaluate(self, eval_args: dict) -> dict:
        return self._bleu(**eval_args)

    def _bleu(self, input: str = None, output: str = None, *args, **kwargs) -> dict:
        if input is None or output is None:
            raise ValueError("Both input (reference) and output must be provided for BLEU.")
        reference = self._tokenize(input)
        candidate = self._tokenize(output)

        result = {"bleu": 0.0, "brevity_penalty": 0.0}
        for n in range(1, self.max_n + 1):
            result[f"precision_{n}"] = 0.0
        if not reference or not candidate:
            return result

        weights = self.weights or [1.0 / self.max_n] * self.max_n

        log_precision_sum = 0.0
        usable_orders = 0
        for n in range(1, self.max_n + 1):
            cand_ngrams = _ngram_counts(candidate, n)
            ref_ngrams = _ngram_counts(reference, n)
            total = sum(cand_ngrams.values())
            overlap = sum(min(cnt, ref_ngrams.get(ng, 0)) for ng, cnt in cand_ngrams.items())
            if total == 0:
                precision = 0.0
            elif overlap == 0:
                precision = (_EPSILON / total) if self.smooth else 0.0
            else:
                precision = overlap / total
            result[f"precision_{n}"] = round(precision, 6)
            weight = weights[n - 1] if n - 1 < len(weights) else 0.0
            if precision > 0:
                log_precision_sum += weight * math.log(precision)
                usable_orders += 1

        # Brevity penalty: penalise candidates shorter than the reference.
        c, r = len(candidate), len(reference)
        brevity_penalty = 1.0 if c > r else math.exp(1 - r / c)
        result["brevity_penalty"] = round(brevity_penalty, 6)

        if usable_orders == 0:
            return result
        result["bleu"] = round(brevity_penalty * math.exp(log_precision_sum), 6)
        return result
