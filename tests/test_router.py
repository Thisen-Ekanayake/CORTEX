"""Tests for cortex/router.py — TFIDFRouter, route_query, and execute()."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from cortex.router import Route, TFIDFRouter, execute


# ── Helpers ───────────────────────────────────────────────────────────────────

def _class_probs(winner: int) -> list[float]:
    """Build a 5-element probability vector with all weight on *winner*."""
    p = [0.0] * 5
    p[winner] = 1.0
    return p


def _make_tfidf_router(predicted_class: int = 0) -> TFIDFRouter:
    """Return a TFIDFRouter backed by configured mock sklearn objects."""
    probs = _class_probs(predicted_class)
    mock_vec = MagicMock()
    mock_clf = MagicMock()
    mock_clf.classes_ = [0, 1, 2, 3, 4]
    mock_clf.predict_proba.return_value = np.array([probs])
    mock_clf.predict.return_value = np.array([predicted_class])

    with patch("cortex.router.os.path.exists", return_value=True), \
         patch("cortex.router.joblib.load", side_effect=[mock_vec, mock_clf]):
        return TFIDFRouter("fake/model")


# ── TFIDFRouter ───────────────────────────────────────────────────────────────

class TestTFIDFRouterInit:
    def test_raises_for_missing_vectorizer(self):
        with patch("cortex.router.os.path.exists", side_effect=[False, True]):
            with pytest.raises(FileNotFoundError, match="vectorizer"):
                TFIDFRouter("fake/model")

    def test_raises_for_missing_classifier(self):
        with patch("cortex.router.os.path.exists", side_effect=[True, False]):
            with pytest.raises(FileNotFoundError, match="Classifier"):
                TFIDFRouter("fake/model")

    def test_raises_for_unknown_class_ids(self):
        mock_vec = MagicMock()
        mock_clf = MagicMock()
        mock_clf.classes_ = [0, 1, 2, 3, 99]   # 99 is not in id_to_route

        with patch("cortex.router.os.path.exists", return_value=True), \
             patch("cortex.router.joblib.load", side_effect=[mock_vec, mock_clf]):
            with pytest.raises(ValueError, match="Unknown class IDs"):
                TFIDFRouter("fake/model")


class TestTFIDFRouterRouteQuery:
    def test_returns_route_enum_and_score_dict(self):
        router = _make_tfidf_router(predicted_class=0)   # CHAT
        route, scores = router.route_query("hello")
        assert isinstance(route, Route)
        assert set(scores.keys()) == {"chat", "meta", "rag", "rag_doc", "rag_img"}
        assert all(isinstance(v, float) for v in scores.values())

    def test_scores_sum_to_one(self):
        router = _make_tfidf_router(predicted_class=2)   # RAG
        _, scores = router.route_query("find the document")
        assert abs(sum(scores.values()) - 1.0) < 1e-6

    @pytest.mark.parametrize("class_id,expected_route", [
        (0, Route.CHAT),
        (1, Route.META),
        (2, Route.RAG),
        (3, Route.RAG_DOC),
        (4, Route.RAG_IMG),
    ])
    def test_class_id_maps_to_correct_route(self, class_id, expected_route):
        router = _make_tfidf_router(predicted_class=class_id)
        route, _ = router.route_query("test query")
        assert route == expected_route


# ── execute() ─────────────────────────────────────────────────────────────────

class TestExecuteForcedRoute:
    """execute() with an explicit route skips the classifier entirely."""

    def test_forced_rag_doc_calls_run_rag_mode_document(self):
        with patch("cortex.router.run_rag_mode", return_value="answer") as mock_fn:
            result, route, _ = execute("query", route=Route.RAG_DOC)
        mock_fn.assert_called_once_with("query", mode="document", callbacks=None)
        assert route == Route.RAG_DOC
        assert result == "answer"

    def test_forced_rag_img_calls_run_rag_mode_image(self):
        with patch("cortex.router.run_rag_mode", return_value="img results") as mock_fn:
            result, route, _ = execute("query", route=Route.RAG_IMG)
        mock_fn.assert_called_once_with("query", mode="image", callbacks=None)
        assert route == Route.RAG_IMG

    def test_forced_meta_calls_run_meta(self):
        with patch("cortex.router.run_meta", return_value="meta answer") as mock_fn:
            result, route, _ = execute("what are you?", route=Route.META)
        mock_fn.assert_called_once()
        assert route == Route.META

    def test_forced_chat_calls_run_chat(self):
        with patch("cortex.router.run_chat", return_value="chat answer") as mock_fn:
            result, route, _ = execute("hello", route=Route.CHAT)
        mock_fn.assert_called_once()
        assert route == Route.CHAT


class TestExecuteAutoRouted:
    """execute() without a forced route uses the classifier + fallback logic."""

    def _scores(self, **overrides: float) -> dict:
        base = {"chat": 0.0, "meta": 0.0, "rag": 0.0, "rag_doc": 0.0, "rag_img": 0.0}
        base.update(overrides)
        return base

    def test_low_rag_confidence_falls_back_to_chat(self):
        scores = self._scores(rag=0.1, rag_doc=0.1, rag_img=0.1, chat=0.7)
        with patch("cortex.router.route_query", return_value=(Route.RAG, scores)), \
             patch("cortex.router.run_chat", return_value="fallback") as mock_chat:
            _, route, _ = execute("some query", confidence_threshold=0.5)
        assert route == Route.CHAT
        mock_chat.assert_called_once()

    def test_high_rag_confidence_calls_run_rag_mode(self):
        scores = self._scores(rag_doc=0.9, chat=0.1)
        with patch("cortex.router.route_query", return_value=(Route.RAG_DOC, scores)), \
             patch("cortex.router.run_rag_mode", return_value="rag answer") as mock_rag:
            result, route, _ = execute("explain this document")
        assert route == Route.RAG_DOC
        mock_rag.assert_called_once()

    def test_rag_none_falls_back_to_chat(self):
        """When RAG retrieval returns nothing, CHAT is used as fallback."""
        scores = self._scores(rag_doc=0.8, chat=0.2)
        with patch("cortex.router.route_query", return_value=(Route.RAG_DOC, scores)), \
             patch("cortex.router.run_rag_mode", return_value=None), \
             patch("cortex.router.run_chat", return_value="chat fallback") as mock_chat:
            _, route, _ = execute("query with empty DB")
        assert route == Route.CHAT
        mock_chat.assert_called_once()

    def test_high_meta_confidence_calls_run_meta(self):
        scores = self._scores(meta=0.9, chat=0.1)
        with patch("cortex.router.route_query", return_value=(Route.META, scores)), \
             patch("cortex.router.run_meta", return_value="meta answer") as mock_meta:
            _, route, _ = execute("what can you do?")
        assert route == Route.META
        mock_meta.assert_called_once()

    def test_low_meta_confidence_falls_back_to_chat(self):
        scores = self._scores(meta=0.3, chat=0.7)
        with patch("cortex.router.route_query", return_value=(Route.META, scores)), \
             patch("cortex.router.run_chat", return_value="chat") as mock_chat:
            _, route, _ = execute("vague question", confidence_threshold=0.5)
        assert route == Route.CHAT
        mock_chat.assert_called_once()

    def test_chat_route_always_calls_run_chat(self):
        scores = self._scores(chat=0.95)
        with patch("cortex.router.route_query", return_value=(Route.CHAT, scores)), \
             patch("cortex.router.run_chat", return_value="hello!") as mock_chat:
            _, route, _ = execute("hi there")
        assert route == Route.CHAT
        mock_chat.assert_called_once()
