"""
Regression tests for issue #456: SentenceComparer must not download the
HuggingFace model at instantiation time (which fails in network-restricted
environments such as the GitHub Coding Agent sandbox).
"""
from unittest.mock import MagicMock, patch


def test_sentence_comparer_instantiation_does_not_load_model():
    """Instantiating SentenceComparer must not trigger a network call."""
    with patch("monocle_test_tools.comparer.sentense_comparer.SentenceTransformer") as mock_st:
        from monocle_test_tools.comparer.sentense_comparer import SentenceComparer
        sc = SentenceComparer()
        mock_st.assert_not_called()


def test_sentence_comparer_loads_model_on_first_compare():
    """The model is loaded the first time compare() is called, not before."""
    mock_model = MagicMock()
    mock_model.encode.return_value = MagicMock(
        __getitem__=lambda self, i: MagicMock(),
    )

    import numpy as np
    vec = np.array([1.0, 0.0], dtype=np.float32)
    mock_model.encode.return_value = np.array([vec, vec])

    with patch("monocle_test_tools.comparer.sentense_comparer.SentenceTransformer", return_value=mock_model) as mock_st:
        from monocle_test_tools.comparer.sentense_comparer import SentenceComparer
        sc = SentenceComparer()
        mock_st.assert_not_called()

        result = sc.compare("hello world", "hello world")
        assert result is True
