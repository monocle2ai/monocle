import logging
from monocle_test_tools.evals.base_eval import BaseEval
from pydantic import Field

logger = logging.getLogger(__name__)

class KeywordPresenceEval(BaseEval):
    """
    Non-LLM evaluator that checks for the presence of required keywords and the
    absence of forbidden keywords in the output.

    eval_options:
        required_keywords (list[str]): Keywords that should appear in the output.
        forbidden_keywords (list[str]): Keywords that should NOT appear in the output.
        case_sensitive (bool): If True, matching is case-sensitive. Default False.
    """
    required_keywords: list = Field(default_factory=list, description="Keywords expected in the output.")
    forbidden_keywords: list = Field(default_factory=list, description="Keywords that must be absent.")
    case_sensitive: bool = Field(default=False, description="Case-sensitive matching when True.")

    def __init__(self, **data):
        super().__init__(**data)
        self.required_keywords = self.eval_options.get("required_keywords", self.required_keywords)
        self.forbidden_keywords = self.eval_options.get("forbidden_keywords", self.forbidden_keywords)
        self.case_sensitive = self.eval_options.get("case_sensitive", self.case_sensitive)

    def evaluate(self, eval_args: dict) -> dict:
        return self._keyword_presence(**eval_args)

    def _keyword_presence(self, output: str = None, *args, **kwargs) -> dict:
        if output is None:
            raise ValueError("Output must be provided for keyword presence checking.")
        text = output if isinstance(output, str) else str(output)
        if not self.case_sensitive:
            text = text.lower()

        def _present(keyword: str) -> bool:
            kw = keyword if self.case_sensitive else keyword.lower()
            return kw in text

        required = list(self.required_keywords or [])
        forbidden = list(self.forbidden_keywords or [])

        found_required = sum(1 for kw in required if _present(kw))
        required_coverage = (found_required / len(required)) if required else 1.0

        found_forbidden = sum(1 for kw in forbidden if _present(kw))
        forbidden_absent = 1.0 if found_forbidden == 0 else 0.0

        return {
            "required_coverage": required_coverage,
            "forbidden_absent": forbidden_absent,
            "missing_required": float(len(required) - found_required),
            "forbidden_found": float(found_forbidden),
        }
