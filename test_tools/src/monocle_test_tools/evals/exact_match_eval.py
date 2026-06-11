import logging
import re
import string
from monocle_test_tools.evals.base_eval import BaseEval
from pydantic import Field

logger = logging.getLogger(__name__)

class ExactMatchEval(BaseEval):
    """
    Non-LLM evaluator that checks whether the output exactly matches a reference
    (the input) after optional normalization. Useful for deterministic, ground-truth
    style assertions (classification labels, canonical answers, etc.).

    eval_options:
        normalize_whitespace (bool): Collapse runs of whitespace and strip. Default True.
        ignore_case (bool): Lowercase both sides before comparing. Default True.
        ignore_punctuation (bool): Strip punctuation before comparing. Default False.
    """
    normalize_whitespace: bool = Field(default=True, description="Collapse and strip whitespace before comparing.")
    ignore_case: bool = Field(default=True, description="Lowercase both sides before comparing.")
    ignore_punctuation: bool = Field(default=False, description="Strip punctuation before comparing.")

    def __init__(self, **data):
        super().__init__(**data)
        self.normalize_whitespace = self.eval_options.get("normalize_whitespace", self.normalize_whitespace)
        self.ignore_case = self.eval_options.get("ignore_case", self.ignore_case)
        self.ignore_punctuation = self.eval_options.get("ignore_punctuation", self.ignore_punctuation)

    def evaluate(self, eval_args: dict) -> dict:
        return self._exact_match(**eval_args)

    def _normalize(self, value: str) -> str:
        text = value if isinstance(value, str) else str(value)
        if self.ignore_case:
            text = text.lower()
        if self.ignore_punctuation:
            text = text.translate(str.maketrans("", "", string.punctuation))
        if self.normalize_whitespace:
            text = re.sub(r"\s+", " ", text).strip()
        return text

    def _exact_match(self, input: str = None, output: str = None, *args, **kwargs) -> dict:
        if input is None or output is None:
            raise ValueError("Both input (reference) and output must be provided for exact match.")
        is_match = self._normalize(input) == self._normalize(output)
        return {"exact_match": 1.0 if is_match else 0.0}
