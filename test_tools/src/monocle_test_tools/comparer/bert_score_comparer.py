import logging
from monocle_test_tools.comparer.base_comparer import BaseComparer

from monocle_test_tools.evals.base_eval import BaseEval
from monocle_test_tools.evals.bert_eval import BertScorerEval

logger = logging.getLogger(__name__)
BERT_ACCEPTABLE_F1_THRESHOLD:float = 0.7
BERT_ACCEPTABLE_PRECISION_THRESHOLD:float = 0.7
BERT_ACCEPTABLE_RECALL_THRESHOLD:float = 0.7
MODEL_TYPE:str = 'bert-base-uncased'

class BertScoreComparer(BaseComparer):
    bert_scorer_eval: BaseEval

    bert_scorer_acceptible_f1_threshold: float = BERT_ACCEPTABLE_F1_THRESHOLD
    bert_scorer_precision_threshold: float = BERT_ACCEPTABLE_PRECISION_THRESHOLD
    bert_scorer_recall_threshold: float = BERT_ACCEPTABLE_RECALL_THRESHOLD
    bert_scorer_eval: BaseEval = None

    def __init__(self, **data):
        super().__init__(**data)
        self.bert_scorer_eval = BertScorerEval()

    def compare(self, expected: str, actual: str) -> bool:
        if expected == actual:
            return True
        if expected is None or actual is None:
            return False
        return self._bert_score_validation(expected, actual)

    def _bert_score_validation(self, expected_response: str, actual_response: str) -> bool:
        """
         Calculate BERTScore between expected and actual responses.
         
         Args:
             expected_response (str): The expected response string.
             actual_response (str): The actual response string.
        """
        eval_args = {
            "input": expected_response,
            "output": actual_response
        }
        scores = self.bert_scorer_eval.evaluate(eval_args=eval_args)
        if scores["F1"] < self.bert_scorer_acceptible_f1_threshold or scores["Recall"] < self.bert_scorer_recall_threshold or scores["Precision"] < self.bert_scorer_precision_threshold:
            logger.debug(f"Output does not match expected. Precision: {scores['Precision']:.4f}, Recall: {scores['Recall']:.4f}, F1: {scores['F1']:.4f}")
            valid_response = False
        else:
            valid_response = True
        return valid_response
