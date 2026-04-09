"""Tests for cortex/query.py — document formatting, source retrieval, RAG modes."""

from unittest.mock import MagicMock, patch

import pytest


def _make_doc(content: str, source: str = "test.pdf", page: int = 1) -> MagicMock:
    doc = MagicMock()
    doc.page_content = content
    doc.metadata = {"source": source, "page": page}
    return doc


class TestFormatDocs:
    def test_empty_list_returns_empty_string(self):
        from cortex.query import format_docs
        assert format_docs([]) == ""

    def test_includes_source_and_page(self):
        from cortex.query import format_docs
        doc = _make_doc("Some content", source="annual_report.pdf", page=7)
        result = format_docs([doc])
        assert "annual_report.pdf" in result
        assert "page 7" in result
        assert "Some content" in result

    def test_multiple_docs_separated_by_double_newline(self):
        from cortex.query import format_docs
        docs = [
            _make_doc("first chunk", source="a.pdf", page=1),
            _make_doc("second chunk", source="b.pdf", page=2),
        ]
        result = format_docs(docs)
        assert "\n\n" in result
        assert "first chunk" in result
        assert "second chunk" in result

    def test_missing_source_metadata_uses_unknown(self):
        from cortex.query import format_docs
        doc = MagicMock()
        doc.page_content = "content"
        doc.metadata = {}   # no source, no page
        result = format_docs([doc])
        assert "unknown" in result
        assert "N/A" in result

    def test_strips_whitespace_from_page_content(self):
        from cortex.query import format_docs
        doc = _make_doc("  lots of whitespace   ")
        result = format_docs([doc])
        assert "lots of whitespace" in result


class TestRunRagModeImage:
    def test_import_error_returns_none(self):
        from cortex.query import run_rag_mode
        with patch.dict("sys.modules", {"image_search.realtime_retrieval": None}):
            result = run_rag_mode("find images", mode="image")
        assert result is None

    def test_search_exception_returns_none(self):
        from cortex.query import run_rag_mode
        mock_mod = MagicMock()
        mock_mod.search_and_retrieve.side_effect = RuntimeError("network error")
        with patch.dict("sys.modules", {"image_search.realtime_retrieval": mock_mod}):
            result = run_rag_mode("find images", mode="image")
        assert result is None

    def test_empty_results_returns_none(self):
        from cortex.query import run_rag_mode
        mock_mod = MagicMock()
        mock_mod.search_and_retrieve.return_value = []
        with patch.dict("sys.modules", {"image_search.realtime_retrieval": mock_mod}):
            result = run_rag_mode("find images", mode="image")
        assert result is None

    def test_returns_formatted_image_paths(self):
        from cortex.query import run_rag_mode
        mock_mod = MagicMock()
        mock_mod.search_and_retrieve.return_value = [
            ("imgs/cat.jpg", 0.95),
            ("imgs/dog.jpg", 0.87),
        ]
        with patch.dict("sys.modules", {"image_search.realtime_retrieval": mock_mod}):
            result = run_rag_mode("show cats", mode="image")
        assert result is not None
        assert "cat.jpg" in result
        assert "0.950" in result
        assert "dog.jpg" in result


class TestGetSources:
    def test_returns_source_strings(self):
        from cortex.query import get_sources
        doc = _make_doc("content", source="manual.pdf", page=3)
        with patch("cortex.query.retrieve_docs", return_value=[doc]):
            sources = get_sources("some query")
        assert len(sources) == 1
        assert "manual.pdf" in sources[0]
        assert "page 3" in sources[0]

    def test_empty_retrieval_returns_empty_list(self):
        from cortex.query import get_sources
        with patch("cortex.query.retrieve_docs", return_value=[]):
            sources = get_sources("unknown query")
        assert sources == []

    def test_multiple_docs_all_returned(self):
        from cortex.query import get_sources
        docs = [_make_doc("c", source=f"doc{i}.pdf", page=i) for i in range(4)]
        with patch("cortex.query.retrieve_docs", return_value=docs):
            sources = get_sources("broad query")
        assert len(sources) == 4
