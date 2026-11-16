"""The classifier that scores text for prompt injection. It runs on this machine, so no text leaves it."""
import hashlib
import importlib.util
import logging
import threading
from collections import OrderedDict
from functools import lru_cache
from typing import List, Optional

logger = logging.getLogger(__name__)

DEFAULT_CLASSIFIER_MODEL = "protectai/deberta-v3-base-prompt-injection-v2"
INJECTION_LABEL = "INJECTION"
LONGEST_TEXT_CLASSIFIED = 20_000
WINDOW_OVERLAP = 64
SCORES_REMEMBERED = 1000
CLASSIFIER_PACKAGES = ("transformers", "torch")
INSTALL_COMMAND = "pip install 'monocle_apptrace[prompt_injection]'"

scores_remembered: "OrderedDict[tuple, float]" = OrderedDict()
one_text_at_a_time = threading.Lock()


def require_classifier_packages(instruction_name: str) -> None:
    """Raise ImportError, with the command that installs them, if the packages the classifier runs on are missing."""
    missing = [package for package in CLASSIFIER_PACKAGES if importlib.util.find_spec(package) is None]
    if missing:
        raise ImportError(f"Instruction '{instruction_name}' sets stop_on_prompt_injection, which needs "
                          f"{' and '.join(missing)}. Install with: {INSTALL_COMMAND}")


def injection_score(text: str, model_name: str) -> float:
    """How sure the classifier is, from 0 to 1, that a text holds a prompt injection.

    Only the first LONGEST_TEXT_CLASSIFIED characters of a text are classified. The score is remembered
    under a hash of the text, so a text seen again is not classified again and is not kept in memory.
    Returns 0 when the classifier cannot be used.
    """
    classifier = classifier_named(model_name or DEFAULT_CLASSIFIER_MODEL)
    if classifier is None:
        return 0.0
    classified = text[:LONGEST_TEXT_CLASSIFIED].encode("utf-8", errors="replace")
    remembered_as = (model_name, hashlib.sha256(classified).digest())
    with one_text_at_a_time:
        if remembered_as not in scores_remembered:
            remember(remembered_as, classifier.injection_score_of(classified.decode("utf-8")))
        return scores_remembered[remembered_as]


def remember(remembered_as: tuple, score: float) -> None:
    scores_remembered[remembered_as] = score
    if len(scores_remembered) > SCORES_REMEMBERED:
        scores_remembered.popitem(last=False)


@lru_cache(maxsize=None)
def classifier_named(model_name: str) -> Optional["LocalClassifier"]:
    """The classifier for a model name, or None if it cannot be used.

    It is loaded the first time it is asked for and kept for the life of the process. One that cannot
    be loaded is not tried again, and a warning says so once.
    """
    try:
        return LocalClassifier(model_name)
    except Exception as e:
        logger.warning("Classifier model '%s' cannot be used, so model calls are not checked for prompt "
                       "injection: %s", model_name, e)
        return None


class LocalClassifier:
    """A Hugging Face text-classification model on this machine's CPU, with an INJECTION label."""

    def __init__(self, model_name: str):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.tokenizer = from_cache_or_hub(AutoTokenizer, model_name)
        self.model = from_cache_or_hub(AutoModelForSequenceClassification, model_name)
        if INJECTION_LABEL not in self.model.config.label2id:
            raise ValueError(f"the model has no {INJECTION_LABEL} label")
        self.injection_label = self.model.config.label2id[INJECTION_LABEL]
        self.window_size = min(self.tokenizer.model_max_length, self.model.config.max_position_embeddings)

    def injection_score_of(self, text: str) -> float:
        """The highest score among the windows of a text, so text longer than the model reads at once is not cut off."""
        return max(self.injection_score_of_window(window) for window in self.windows_of(text))

    def windows_of(self, text: str) -> List[List[int]]:
        """The text as tokens, in overlapping windows that the model can each read whole."""
        return self.tokenizer(text, truncation=True, max_length=self.window_size, stride=WINDOW_OVERLAP,
                              return_overflowing_tokens=True)["input_ids"]

    def injection_score_of_window(self, window: List[int]) -> float:
        import torch
        with torch.no_grad():
            scores = self.model(input_ids=torch.tensor([window])).logits.softmax(dim=-1)
        return scores[0, self.injection_label].item()


def from_cache_or_hub(kind, model_name: str):
    """A tokenizer or model from the local Hugging Face cache, or downloaded from the hub when it is not there yet."""
    try:
        return kind.from_pretrained(model_name, local_files_only=True)
    except OSError:
        return kind.from_pretrained(model_name)
