import logging
import re
from monocle_test_tools.evals.base_eval import BaseEval
from pydantic import Field

logger = logging.getLogger(__name__)

class RegexMatchEval(BaseEval):
    """
    Non-LLM evaluator that checks whether the output matches a regular expression.

    eval_options:
        pattern (str): The regular expression to search for. Required.
        ignore_case (bool): If True, matching is case-insensitive. Default False.
        full_match (bool): If True, the whole output must match the pattern
            (re.fullmatch) instead of containing a match (re.search). Default False.
    """
    pattern: str = Field(default="", description="Regular expression to match against the output.")
    ignore_case: bool = Field(default=False, description="Case-insensitive matching when True.")
    full_match: bool = Field(default=False, description="Require the whole output to match when True.")

    def __init__(self, **data):
        super().__init__(**data)
        self.pattern = self.eval_options.get("pattern", self.pattern)
        self.ignore_case = self.eval_options.get("ignore_case", self.ignore_case)
        self.full_match = self.eval_options.get("full_match", self.full_match)

    def evaluate(self, eval_args: dict) -> dict:
        return self._regex_match(**eval_args)

    def _regex_match(self, output: str = None, *args, **kwargs) -> dict:
        if output is None:
            raise ValueError("Output must be provided for regex matching.")
        if not self.pattern:
            raise ValueError("A 'pattern' must be provided in eval_options for RegexMatchEval.")
        flags = re.IGNORECASE if self.ignore_case else 0
        compiled = re.compile(self.pattern, flags)
        text = output if isinstance(output, str) else str(output)
        if self.full_match:
            matched = compiled.fullmatch(text) is not None
            match_count = 1 if matched else 0
        else:
            matches = compiled.findall(text)
            match_count = len(matches)
            matched = match_count > 0
        return {"match": 1.0 if matched else 0.0, "match_count": float(match_count)}
