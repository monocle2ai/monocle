from typing import Optional
from pydantic import BaseModel 
class BaseEval(BaseModel):
    eval_options: Optional[dict] = {}
    def __init__(self, **data):
        super().__init__(**data)

    def get_eval(self, eval_name:str, eval_args: dict) -> dict:
        raise NotImplementedError