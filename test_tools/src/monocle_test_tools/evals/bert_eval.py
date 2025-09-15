import logging
from monocle_test_tools.evals.base_eval import BaseEval
from bert_score import BERTScorer
from pydantic import Field

logger = logging.getLogger(__name__)
MODEL_TYPE:str = 'bert-base-uncased'

class BertScorerEval(BaseEval):
    model_type: str = Field(default=MODEL_TYPE, description="BERT model type to use for scoring.")

    def __init__(self, **data):
        super().__init__(**data)
        self.model_type = self.eval_options.get("model_type", MODEL_TYPE)

    def get_eval(self, eval_name: str, eval_args: dict) -> dict:
        # Implement default evaluation logic here
        if eval_name == "bert_score":
            return self._bert_score_validation(**eval_args)
        else:
            raise ValueError(f"Unsupported eval: {eval_name}.")

    
    def _bert_score_validation(self, expected_response: str, actual_response: str) -> dict:
        """
         Calculate BERTScore between expected and actual responses.
         
         Args:
             expected_response (str): The expected response string.
             actual_response (str): The actual response string.
        """
        scorer = BERTScorer(model_type=self.model_type, use_fast_tokenizer=True)
        precision_score, recall_score, F1_score = scorer.score([actual_response], [expected_response])
        return {"Precision": precision_score.mean(), "Recall": recall_score.mean(), "F1": F1_score.mean()}
