"""Tests for image_search/realtime_retrieval.py — download_images error handling."""

from unittest.mock import MagicMock, patch

import pytest
import requests


class TestDownloadImages:
    """
    Tests focus on the error-handling behaviour of download_images().
    The function must log and skip failures rather than crashing.
    """

    def test_empty_url_list_returns_empty(self, tmp_path):
        with patch("image_search.realtime_retrieval.IMAGE_CACHE_DIR", str(tmp_path)):
            from image_search.realtime_retrieval import download_images
            result = download_images([])
        assert result == []

    def test_request_exception_is_skipped(self, tmp_path):
        with patch("image_search.realtime_retrieval.IMAGE_CACHE_DIR", str(tmp_path)), \
             patch("image_search.realtime_retrieval.requests.get",
                   side_effect=requests.RequestException("timeout")):
            from image_search.realtime_retrieval import download_images
            result = download_images(["http://example.com/img.jpg"])
        assert result == []

    def test_http_error_is_skipped(self, tmp_path):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.HTTPError("404")
        with patch("image_search.realtime_retrieval.IMAGE_CACHE_DIR", str(tmp_path)), \
             patch("image_search.realtime_retrieval.requests.get", return_value=mock_resp):
            from image_search.realtime_retrieval import download_images
            result = download_images(["http://example.com/missing.png"])
        assert result == []

    def test_successful_download_returns_saved_path(self, tmp_path):
        mock_resp = MagicMock()
        mock_resp.content = b"fake image bytes"
        mock_resp.raise_for_status.return_value = None
        with patch("image_search.realtime_retrieval.IMAGE_CACHE_DIR", str(tmp_path)), \
             patch("image_search.realtime_retrieval.requests.get", return_value=mock_resp):
            from image_search.realtime_retrieval import download_images
            result = download_images(["http://example.com/photo.jpg"])
        assert len(result) == 1
        assert result[0].endswith(".jpg")

    def test_mixed_success_and_failure_returns_only_successes(self, tmp_path):
        good_resp = MagicMock()
        good_resp.content = b"data"
        good_resp.raise_for_status.return_value = None

        def _get(url, timeout):
            if "good" in url:
                return good_resp
            raise requests.RequestException("fail")

        with patch("image_search.realtime_retrieval.IMAGE_CACHE_DIR", str(tmp_path)), \
             patch("image_search.realtime_retrieval.requests.get", side_effect=_get):
            from image_search.realtime_retrieval import download_images
            result = download_images([
                "http://example.com/bad.jpg",
                "http://example.com/good.png",
                "http://example.com/also_bad.jpg",
            ])
        assert len(result) == 1

    def test_os_error_on_write_is_skipped(self, tmp_path):
        mock_resp = MagicMock()
        mock_resp.content = b"data"
        mock_resp.raise_for_status.return_value = None
        with patch("image_search.realtime_retrieval.IMAGE_CACHE_DIR", str(tmp_path)), \
             patch("image_search.realtime_retrieval.requests.get", return_value=mock_resp), \
             patch("builtins.open", side_effect=OSError("disk full")):
            from image_search.realtime_retrieval import download_images
            result = download_images(["http://example.com/img.jpg"])
        assert result == []
