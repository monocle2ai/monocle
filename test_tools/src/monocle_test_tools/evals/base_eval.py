from typing import Optional, Union, Tuple
from opentelemetry.sdk.trace import Span
from pydantic import BaseModel

class BaseEval(BaseModel):
    eval_options: Optional[dict] = {}
    def __init__(self, **data):       
        super().__init__(**data)

    def evaluate(self, filtered_spans:Optional[list[Span]] = [],  eval_name:Optional[str] = "", fact_name: Optional[str] = "traces", eval_args: dict = {}, template: Optional[dict] = None) -> Union[str,dict]:
        raise NotImplementedError
    
    def cleanup(self):
        """Optional cleanup hook called at test end. Override if needed."""
        pass

    @classmethod
    def classify_eval_input(cls, name_or_path: str) -> Tuple[str, str]:
        """Classify an eval input as builtin or custom.

        Returns ``(eval_type, value)`` where eval_type is ``"builtin"`` or ``"custom"``.
        Default: path-like strings (.json / path separator) → custom, else → builtin.
        Subclasses (e.g. OkahuEval) may override.
        """
        is_path_like = (
            name_or_path.endswith(".json")
            or "/" in name_or_path
            or name_or_path.startswith("./")
            or name_or_path.startswith("../")
        )
        return ("custom" if is_path_like else "builtin"), name_or_path