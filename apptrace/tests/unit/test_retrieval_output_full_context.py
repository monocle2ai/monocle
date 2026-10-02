"""Retrieval spans must carry the full retrieved context (every document, not a 100-char stub)."""
from types import SimpleNamespace

import pytest

from monocle_apptrace.instrumentation.common.utils import (
    DEFAULT_RETRIEVAL_OUTPUT_MAX_CHARS,
    MONOCLE_RETRIEVAL_OUTPUT_MAX_CHARS_ENV,
    RETRIEVAL_DOCUMENT_SEPARATOR,
    format_retrieved_documents,
)
from monocle_apptrace.instrumentation.metamodel.haystack import _helper as haystack_helper
from monocle_apptrace.instrumentation.metamodel.langchain import _helper as langchain_helper
from monocle_apptrace.instrumentation.metamodel.llamaindex import _helper as llamaindex_helper

DOCS = ["first document " * 20, "second document " * 20, "third document " * 20]
EXPECTED = RETRIEVAL_DOCUMENT_SEPARATOR.join(DOCS)


@pytest.fixture(autouse=True)
def _clear_limit_env(monkeypatch):
    monkeypatch.delenv(MONOCLE_RETRIEVAL_OUTPUT_MAX_CHARS_ENV, raising=False)


def test_llamaindex_records_every_node_in_full():
    nodes = [SimpleNamespace(text=d) for d in DOCS]
    assert llamaindex_helper.update_output_span_events(nodes) == EXPECTED


def test_langchain_records_every_document_in_full():
    docs = [SimpleNamespace(page_content=d) for d in DOCS]
    assert langchain_helper.update_output_span_events(docs) == EXPECTED


def test_haystack_records_every_document_in_full():
    result = {"documents": [SimpleNamespace(content=d) for d in DOCS]}
    assert haystack_helper.update_output_span_events(result) == EXPECTED


def test_default_limit_truncates_huge_retrievals():
    output = format_retrieved_documents(["x" * (DEFAULT_RETRIEVAL_OUTPUT_MAX_CHARS + 10)])
    assert output == "x" * DEFAULT_RETRIEVAL_OUTPUT_MAX_CHARS + "..."


def test_limit_is_configurable_via_env(monkeypatch):
    monkeypatch.setenv(MONOCLE_RETRIEVAL_OUTPUT_MAX_CHARS_ENV, "10")
    assert format_retrieved_documents(["abcdefghijklmnop"]) == "abcdefghij..."


@pytest.mark.parametrize("value", ["not-a-number", "0", "-5"])
def test_invalid_env_limit_falls_back_to_default(monkeypatch, value):
    monkeypatch.setenv(MONOCLE_RETRIEVAL_OUTPUT_MAX_CHARS_ENV, value)
    text = "y" * 1000
    assert format_retrieved_documents([text]) == text


def test_none_texts_are_skipped():
    assert format_retrieved_documents(["a", None, "b"]) == "a" + RETRIEVAL_DOCUMENT_SEPARATOR + "b"
