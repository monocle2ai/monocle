import logging
import re
from monocle_test_tools.evals.base_eval import BaseEval

logger = logging.getLogger(__name__)

class ReadabilityEval(BaseEval):
    """
    Non-LLM evaluator that computes deterministic readability metrics for the output:
    the Flesch Reading Ease score and the Flesch-Kincaid Grade Level. Both are derived
    purely from word, sentence and syllable counts (no model involved).

    Returns:
        flesch_reading_ease (float): Higher is easier to read (0-100+ scale).
        flesch_kincaid_grade (float): Approximate US school grade level.
        word_count / sentence_count (float): Raw counts used in the computation.
    """

    def evaluate(self, eval_args: dict) -> dict:
        return self._readability(**eval_args)

    @staticmethod
    def _count_syllables(word: str) -> int:
        word = word.lower()
        word = re.sub(r"[^a-z]", "", word)
        if not word:
            return 0
        vowels = "aeiouy"
        count = 0
        prev_was_vowel = False
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                count += 1
            prev_was_vowel = is_vowel
        # Silent trailing 'e'
        if word.endswith("e") and count > 1:
            count -= 1
        return max(count, 1)

    def _readability(self, output: str = None, *args, **kwargs) -> dict:
        if output is None:
            raise ValueError("Output must be provided for readability scoring.")
        text = output if isinstance(output, str) else str(output)

        sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
        sentence_count = max(len(sentences), 1)
        words = re.findall(r"[A-Za-z0-9']+", text)
        word_count = len(words)
        if word_count == 0:
            return {
                "flesch_reading_ease": 0.0,
                "flesch_kincaid_grade": 0.0,
                "word_count": 0.0,
                "sentence_count": float(sentence_count),
            }
        syllable_count = sum(self._count_syllables(w) for w in words)

        words_per_sentence = word_count / sentence_count
        syllables_per_word = syllable_count / word_count

        flesch_reading_ease = 206.835 - (1.015 * words_per_sentence) - (84.6 * syllables_per_word)
        flesch_kincaid_grade = (0.39 * words_per_sentence) + (11.8 * syllables_per_word) - 15.59

        return {
            "flesch_reading_ease": round(flesch_reading_ease, 2),
            "flesch_kincaid_grade": round(flesch_kincaid_grade, 2),
            "word_count": float(word_count),
            "sentence_count": float(sentence_count),
        }
