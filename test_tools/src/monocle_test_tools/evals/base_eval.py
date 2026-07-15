from typing import Optional, Union
from opentelemetry.sdk.trace import Span
from pydantic import BaseModel, ConfigDict

class BaseEval(BaseModel):
    model_config = ConfigDict(extra='allow')
    eval_options: Optional[dict] = {}
    def __init__(self, **data):       
        super().__init__(**data)

    def evaluate(self, filtered_spans:Optional[list[Span]] = [],  eval_name:Optional[str] = "", fact_name: Optional[str] = "traces", eval_args: dict = {}, template: Optional[dict] = None) -> Union[str,dict]:
        raise NotImplementedError
    
    def cleanup(self):
        """Optional cleanup hook called at test end. Override if needed."""
        pass