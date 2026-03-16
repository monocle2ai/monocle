"""
Pre-download all HuggingFace models needed by monocle_test_tools.

Run this ONCE while network is available (e.g. during CI setup or
after ``pip install monocle_test_tools``), so that model
weights are cached locally.  After that, tests can run fully offline.

Usage:
    python -m monocle_test_tools.download_models
    # or via entry point:
    monocle-download-models
"""

import logging

logger = logging.getLogger(__name__)

# All models that monocle_test_tools needs at runtime
MODELS = {
    "sentence-transformers": "sentence-transformers/all-mpnet-base-v2",
    "bert-score": "bert-base-uncased",
}


def _is_model_cached(model_name: str) -> bool:
    """Check whether *model_name* is already present in the HuggingFace cache."""
    try:
        from huggingface_hub import try_to_load_from_cache
        # A model is cached when at least its config.json is available locally.
        result = try_to_load_from_cache(model_name, "config.json")
        return result is not None and isinstance(result, str)
    except Exception:
        return False


def download_all() -> None:
    """Download and cache all required HuggingFace models.

    Skips models that are already cached.
    """
    print("Checking HuggingFace models for monocle_test_tools...")

    # 1. sentence-transformers model (used by SentenceComparer)
    st_model = MODELS["sentence-transformers"]
    if _is_model_cached(st_model):
        print(f"  [OK] {st_model} already cached.")
    else:
        try:
            from sentence_transformers import SentenceTransformer
            print(f"  [DOWNLOADING] {st_model} ...")
            SentenceTransformer(model_name_or_path=st_model)
            print(f"  [OK] {st_model} cached successfully.")
        except Exception as e:
            print(f"  [FAILED] Failed to download sentence-transformers model: {e}")
            raise

    # 2. bert-score model (used by BertScoreComparer / BertScorerEval)
    bert_model = MODELS["bert-score"]
    if _is_model_cached(bert_model):
        print(f"  [OK] {bert_model} already cached.")
    else:
        try:
            from bert_score import BERTScorer
            print(f"  [DOWNLOADING] {bert_model} ...")
            BERTScorer(model_type=bert_model, use_fast_tokenizer=True)
            print(f"  [OK] {bert_model} cached successfully.")
        except Exception as e:
            print(f"  [FAILED] Failed to download bert-score model: {e}")
            raise

    print("All models ready. Tests can now run offline.")


if __name__ == "__main__":
    download_all()
