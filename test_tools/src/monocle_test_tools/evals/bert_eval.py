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

    def evaluate(self, eval_args: dict) -> dict:
        return self._bert_score_validation(**eval_args)

    def _bert_score_validation(self, input: str, output: str, *args, **kwargs) -> dict:
        """
         Calculate BERTScore between expected and actual responses.
         
         Args:
             input (str): The expected response string.
             output (str): The actual response string.
        """
        if input is None or output is None:
            raise ValueError("Input and output must be provided for BERT scoring.")
        scorer = BERTScorer(model_type=self.model_type, use_fast_tokenizer=True)
        precision_score, recall_score, F1_score = scorer.score([output], [input])
        return {"Precision": precision_score.mean().item(), "Recall": recall_score.mean().item(), "F1": F1_score.mean().item()}
