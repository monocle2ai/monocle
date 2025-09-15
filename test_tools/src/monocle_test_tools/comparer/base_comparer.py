from typing import Optional, Union
from pydantic import BaseModel

class BaseComparer(BaseModel):
    eval_options: Optional[dict] = {}
    def __init__(self, **data):
        super().__init__(**data)

    def compare(self, expected: Union[dict, str], actual: Union[dict, str]) -> bool:
        raise NotImplementedError