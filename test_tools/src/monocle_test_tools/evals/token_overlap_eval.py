import logging
import re
from collections import Counter
from monocle_test_tools.evals.base_eval import BaseEval
from pydantic import Field

logger = logging.getLogger(__name__)

class TokenOverlapEval(BaseEval):
    """
    Non-LLM evaluator that measures lexical token overlap between a reference (input)
    and the output, reporting ROUGE-style precision, recall and F1. This is a purely
    deterministic, bag-of-tokens comparison — no embeddings or model required.

    eval_options:
        ignore_case (bool): Lowercase both sides before tokenizing. Default True.
        token_pattern (str): Regex used to extract tokens. Default word characters.

    Returns:
        precision / recall / f1 (float): Overlap scores in the range [0.0, 1.0].
    """
    ignore_case: bool = Field(default=True, description="Lowercase both sides before tokenizing.")
    token_pattern: str = Field(default=r"[A-Za-z0-9']+", description="Regex used to extract tokens.")

    def __init__(self, **data):
        super().__init__(**data)
        self.ignore_case = self.eval_options.get("ignore_case", self.ignore_case)
        self.token_pattern = self.eval_options.get("token_pattern", self.token_pattern)

    def _tokenize(self, value: str) -> list:
        text = value if isinstance(value, str) else str(value)
        if self.ignore_case:
            text = text.lower()
        return re.findall(self.token_pattern, text)

    def evaluate(self, eval_args: dict) -> dict:
        return self._token_overlap(**eval_args)

    def _token_overlap(self, input: str = None, output: str = None, *args, **kwargs) -> dict:
        if input is None or output is None:
            raise ValueError("Both input (reference) and output must be provided for token overlap.")
        ref_tokens = self._tokenize(input)
        hyp_tokens = self._tokenize(output)
        if not ref_tokens or not hyp_tokens:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        ref_counts = Counter(ref_tokens)
        hyp_counts = Counter(hyp_tokens)
        overlap = sum((ref_counts & hyp_counts).values())

        precision = overlap / len(hyp_tokens)
        recall = overlap / len(ref_tokens)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
