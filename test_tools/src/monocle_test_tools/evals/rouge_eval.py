import logging
import re
from collections import Counter
from monocle_test_tools.evals.base_eval import BaseEval
from pydantic import Field

logger = logging.getLogger(__name__)


def _ngram_counts(tokens: list, n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _lcs_length(a: list, b: list) -> int:
    """Length of the longest common subsequence of two token lists."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for token_a in a:
        curr = [0] * (len(b) + 1)
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[-1]


def _prf(overlap: int, hyp_total: int, ref_total: int) -> tuple:
    precision = overlap / hyp_total if hyp_total else 0.0
    recall = overlap / ref_total if ref_total else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


class RougeEval(BaseEval):
    """
    Non-LLM evaluator that computes ROUGE recall-oriented overlap metrics between a
    reference (``input``) and a candidate (``output``). Supports ROUGE-N (n-gram
    overlap) and ROUGE-L (longest common subsequence). This is a pure, deterministic
    implementation — no model, no network, no external dependency.

    eval_options:
        rouge_types (list[str]): Any of ``"rouge1"``, ``"rouge2"``, ... and
            ``"rougeL"``. Default ``["rouge1", "rouge2", "rougeL"]``.
        ignore_case (bool): Lowercase both sides before tokenizing. Default True.
        token_pattern (str): Regex used to extract tokens. Default word characters.

    Returns (flat, numeric — one set per requested type):
        <type>_p (float): precision, <type>_r (float): recall, <type>_f (float): F1.
        e.g. ``rouge1_f``, ``rouge2_r``, ``rougeL_p``.
    """
    rouge_types: list = Field(default=None, description="ROUGE variants to compute.")
    ignore_case: bool = Field(default=True, description="Lowercase both sides before tokenizing.")
    token_pattern: str = Field(default=r"[A-Za-z0-9']+", description="Regex used to extract tokens.")

    def __init__(self, **data):
        super().__init__(**data)
        self.rouge_types = self.eval_options.get("rouge_types", self.rouge_types) or [
            "rouge1", "rouge2", "rougeL"
        ]
        self.ignore_case = self.eval_options.get("ignore_case", self.ignore_case)
        self.token_pattern = self.eval_options.get("token_pattern", self.token_pattern)

    def _tokenize(self, value: str) -> list:
        text = value if isinstance(value, str) else str(value)
        if self.ignore_case:
            text = text.lower()
        return re.findall(self.token_pattern, text)

    def evaluate(self, eval_args: dict) -> dict:
        return self._rouge(**eval_args)

    def _rouge(self, input: str = None, output: str = None, *args, **kwargs) -> dict:
        if input is None or output is None:
            raise ValueError("Both input (reference) and output must be provided for ROUGE.")
        reference = self._tokenize(input)
        candidate = self._tokenize(output)

        result = {}
        for rouge_type in self.rouge_types:
            if rouge_type.lower() == "rougel":
                overlap = _lcs_length(reference, candidate)
                precision, recall, f1 = _prf(overlap, len(candidate), len(reference))
            else:
                match = re.fullmatch(r"rouge(\d+)", rouge_type.lower())
                if not match:
                    raise ValueError(f"Unsupported rouge_type: {rouge_type!r}")
                n = int(match.group(1))
                ref_ngrams = _ngram_counts(reference, n)
                cand_ngrams = _ngram_counts(candidate, n)
                overlap = sum(min(cnt, cand_ngrams.get(ng, 0)) for ng, cnt in ref_ngrams.items())
                precision, recall, f1 = _prf(overlap, sum(cand_ngrams.values()), sum(ref_ngrams.values()))
            result[f"{rouge_type}_p"] = round(precision, 4)
            result[f"{rouge_type}_r"] = round(recall, 4)
            result[f"{rouge_type}_f"] = round(f1, 4)
        return result
