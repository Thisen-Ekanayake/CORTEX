"""Tests for cortex/ingest.py — document loading and ingestion pipeline."""

from unittest.mock import MagicMock, call, patch

import pytest


def _make_mock_doc(content: str = "test content", source: str = "") -> MagicMock:
    doc = MagicMock()
    doc.page_content = content
    doc.metadata = {}
    if source:
        doc.metadata["source"] = source
    return doc


class TestLoadDocuments:
    def test_skips_unsupported_file_types(self, tmp_path):
        (tmp_path / "notes.odt").write_bytes(b"odt")
        (tmp_path / "image.png").write_bytes(b"png")

        with patch("cortex.ingest.DATA_DIR", str(tmp_path)):
            from cortex.ingest import load_documents
            docs = load_documents()

        assert docs == []

    def test_sets_source_metadata_on_pdf(self, tmp_path):
        (tmp_path / "report.pdf").write_bytes(b"fake-pdf")
        mock_doc = _make_mock_doc()

        with patch("cortex.ingest.DATA_DIR", str(tmp_path)), \
             patch("cortex.ingest.PyPDFLoader") as MockLoader:
            MockLoader.return_value.load.return_value = [mock_doc]
            from cortex.ingest import load_documents
            docs = load_documents()

        assert docs[0].metadata["source"] == "report.pdf"

    def test_regression_loader_called_exactly_once(self, tmp_path):
        """
        Regression: metadata must not be discarded by a second load() call.

        The original bug called loader.load() twice — once to iterate and set
        metadata, then again to extend() the list, throwing the metadata away.
        """
        (tmp_path / "doc.pdf").write_bytes(b"fake-pdf")
        mock_doc = _make_mock_doc()

        with patch("cortex.ingest.DATA_DIR", str(tmp_path)), \
             patch("cortex.ingest.PyPDFLoader") as MockLoader:
            MockLoader.return_value.load.return_value = [mock_doc]
            from cortex.ingest import load_documents
            load_documents()

        MockLoader.return_value.load.assert_called_once()

    def test_loads_txt_files(self, tmp_path):
        (tmp_path / "notes.txt").write_text("hello", encoding="utf-8")
        mock_doc = _make_mock_doc()

        with patch("cortex.ingest.DATA_DIR", str(tmp_path)), \
             patch("cortex.ingest.TextLoader") as MockLoader:
            MockLoader.return_value.load.return_value = [mock_doc]
            from cortex.ingest import load_documents
            docs = load_documents()

        assert len(docs) == 1
        assert docs[0].metadata["source"] == "notes.txt"

    def test_loads_docx_files(self, tmp_path):
        (tmp_path / "memo.docx").write_bytes(b"fake-docx")
        mock_doc = _make_mock_doc()

        with patch("cortex.ingest.DATA_DIR", str(tmp_path)), \
             patch("cortex.ingest.Docx2txtLoader") as MockLoader:
            MockLoader.return_value.load.return_value = [mock_doc]
            from cortex.ingest import load_documents
            docs = load_documents()

        assert len(docs) == 1
        assert docs[0].metadata["source"] == "memo.docx"

    def test_empty_directory_returns_empty_list(self, tmp_path):
        with patch("cortex.ingest.DATA_DIR", str(tmp_path)):
            from cortex.ingest import load_documents
            docs = load_documents()

        assert docs == []


class TestIngest:
    def test_ingest_returns_early_when_no_documents(self, tmp_path, capsys):
        with patch("cortex.ingest.DATA_DIR", str(tmp_path)), \
             patch("cortex.ingest.PERSIST_DIR", str(tmp_path / "db")):
            from cortex.ingest import ingest
            ingest()

        captured = capsys.readouterr()
        assert "No documents found" in captured.out

    def test_ingest_calls_vectorstore_persist(self, tmp_path):
        mock_doc = _make_mock_doc("content")
        mock_chunk = _make_mock_doc("chunked content")
        mock_vectorstore = MagicMock()

        with patch("cortex.ingest.DATA_DIR", str(tmp_path)), \
             patch("cortex.ingest.PERSIST_DIR", str(tmp_path / "db")), \
             patch("cortex.ingest.load_documents", return_value=[mock_doc]), \
             patch("cortex.ingest.RecursiveCharacterTextSplitter") as MockSplitter, \
             patch("cortex.ingest.Chroma") as MockChroma:
            MockSplitter.return_value.split_documents.return_value = [mock_chunk]
            MockChroma.from_documents.return_value = mock_vectorstore

            from cortex.ingest import ingest
            ingest()

        mock_vectorstore.persist.assert_called_once()
