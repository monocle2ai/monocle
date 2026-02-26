from typing import Optional, Union
from opentelemetry.sdk.trace import Span
from pydantic import BaseModel

class BaseEval(BaseModel):
    eval_options: Optional[dict] = {}
    def __init__(self, **data):       
        super().__init__(**data)

    def evaluate(self, filtered_spans:Optional[list[Span]] = [],  eval_name:Optional[str] = "", eval_args: dict = {}) -> Union[str,dict]:
        raise NotImplementedError