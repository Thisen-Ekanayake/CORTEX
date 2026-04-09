"""Tests for cortex/rl_router.py — weight learning, metrics, and persistence."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from cortex.router import Route
from cortex.rl_router import RLRouter, RoutingFeedback


# ── Helpers ───────────────────────────────────────────────────────────────────

_ALL_ROUTES = [r.value for r in Route]

def _scores(**overrides: float) -> dict:
    """Build a complete confidence-score dict, defaulting to uniform distribution."""
    base = {r: 0.0 for r in _ALL_ROUTES}
    base.update(overrides)
    return base


def _make_base_router(predicted: Route = Route.CHAT) -> MagicMock:
    mock = MagicMock()
    s = _scores(**{predicted.value: 0.8})
    mock.route_query.return_value = (predicted, s)
    return mock


@pytest.fixture
def rl(tmp_path) -> RLRouter:
    return RLRouter(
        base_router=_make_base_router(),
        feedback_dir=str(tmp_path),
        learning_rate=0.1,
    )


# ── Initial state ─────────────────────────────────────────────────────────────

class TestInitialState:
    def test_all_weights_start_at_one(self, rl):
        for route in Route:
            assert rl.confidence_weights[route.value] == 1.0

    def test_metrics_start_at_zero(self, rl):
        m = rl.get_metrics()
        assert m["total_predictions"] == 0
        assert m["correct_predictions"] == 0
        assert m["overall_accuracy"] == 0.0


# ── Reward signal ─────────────────────────────────────────────────────────────

class TestRewardSignal:
    def test_correct_prediction_gives_positive_reward(self, rl):
        fb = rl.record_feedback(
            "test", Route.CHAT, Route.CHAT,
            _scores(chat=0.8, meta=0.1, rag=0.05, rag_doc=0.03, rag_img=0.02),
        )
        assert fb.correct is True
        assert fb.reward > 0

    def test_incorrect_prediction_gives_negative_reward(self, rl):
        fb = rl.record_feedback(
            "test", Route.CHAT, Route.META,
            _scores(chat=0.8, meta=0.1, rag=0.05, rag_doc=0.03, rag_img=0.02),
        )
        assert fb.correct is False
        assert fb.reward < 0


# ── Weight updates ────────────────────────────────────────────────────────────

class TestWeightUpdates:
    _s = _scores(chat=0.8, meta=0.1, rag=0.05, rag_doc=0.03, rag_img=0.02)

    def test_correct_prediction_increases_weight(self, rl):
        before = rl.confidence_weights[Route.CHAT.value]
        rl.record_feedback("q", Route.CHAT, Route.CHAT, self._s)
        assert rl.confidence_weights[Route.CHAT.value] > before

    def test_incorrect_prediction_decreases_weight_for_predicted_route(self, rl):
        before = rl.confidence_weights[Route.CHAT.value]
        rl.record_feedback("q", Route.CHAT, Route.META, self._s)
        assert rl.confidence_weights[Route.CHAT.value] < before

    def test_incorrect_prediction_increases_weight_for_actual_route(self, rl):
        before = rl.confidence_weights[Route.META.value]
        rl.record_feedback("q", Route.CHAT, Route.META, self._s)
        assert rl.confidence_weights[Route.META.value] > before

    def test_weight_never_exceeds_max(self, rl):
        rl.confidence_weights[Route.CHAT.value] = 1.999
        rl.record_feedback(
            "q", Route.CHAT, Route.CHAT,
            _scores(chat=1.0),
        )
        assert rl.confidence_weights[Route.CHAT.value] <= 2.0

    def test_weight_never_drops_below_min(self, rl):
        rl.confidence_weights[Route.CHAT.value] = 0.501
        rl.record_feedback(
            "q", Route.CHAT, Route.META,
            _scores(chat=0.9, meta=0.1),
        )
        assert rl.confidence_weights[Route.CHAT.value] >= 0.5


# ── Metrics ───────────────────────────────────────────────────────────────────

class TestMetrics:
    _s = _scores(chat=0.8, meta=0.1, rag=0.05, rag_doc=0.03, rag_img=0.02)

    def test_accuracy_two_correct_one_wrong(self, rl):
        rl.record_feedback("q1", Route.CHAT, Route.CHAT, self._s)
        rl.record_feedback("q2", Route.CHAT, Route.CHAT, self._s)
        rl.record_feedback("q3", Route.CHAT, Route.META, self._s)

        m = rl.get_metrics()
        assert m["total_predictions"] == 3
        assert m["correct_predictions"] == 2
        assert abs(m["overall_accuracy"] - 2 / 3) < 1e-9

    def test_reset_clears_all_stats_and_weights(self, rl):
        rl.record_feedback("q", Route.CHAT, Route.CHAT, self._s)
        rl.reset_learning()

        m = rl.get_metrics()
        assert m["total_predictions"] == 0
        assert m["correct_predictions"] == 0
        for route in Route:
            assert rl.confidence_weights[route.value] == 1.0


# ── Persistence ───────────────────────────────────────────────────────────────

class TestPersistence:
    _s = _scores(chat=0.8, meta=0.1, rag=0.05, rag_doc=0.03, rag_img=0.02)

    def test_feedback_written_to_jsonl(self, rl, tmp_path):
        rl.record_feedback("my query", Route.CHAT, Route.CHAT, self._s)

        lines = (tmp_path / "feedback.jsonl").read_text().strip().splitlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["query"] == "my query"
        assert data["correct"] is True

    def test_feedback_loaded_by_new_instance(self, tmp_path):
        base = _make_base_router()
        r1 = RLRouter(base_router=base, feedback_dir=str(tmp_path))
        r1.record_feedback("persistent query", Route.CHAT, Route.CHAT, self._s)

        r2 = RLRouter(base_router=base, feedback_dir=str(tmp_path))
        assert any(fb.query == "persistent query" for fb in r2.feedback_history)

    def test_metrics_persisted_and_reloaded(self, tmp_path):
        base = _make_base_router()
        r1 = RLRouter(base_router=base, feedback_dir=str(tmp_path))
        r1.record_feedback("q", Route.CHAT, Route.CHAT, self._s)
        r1.record_feedback("q", Route.CHAT, Route.META, self._s)

        r2 = RLRouter(base_router=base, feedback_dir=str(tmp_path))
        assert r2.total_predictions == 2
        assert r2.correct_predictions == 1

    def test_get_recent_feedback_returns_last_n(self, rl):
        for i in range(6):
            rl.record_feedback(f"query {i}", Route.CHAT, Route.CHAT, self._s)

        recent = rl.get_recent_feedback(n=3)
        assert len(recent) == 3
        assert recent[-1].query == "query 5"


# ── Routing with RL adjustments ───────────────────────────────────────────────

class TestRLRouting:
    def test_route_query_returns_route_and_adjusted_scores(self):
        base = _make_base_router(predicted=Route.META)
        rl = RLRouter(base_router=base, feedback_dir="/tmp/rl_test_routing")
        route, scores = rl.route_query("test")
        assert isinstance(route, Route)
        assert set(scores.keys()) == set(_ALL_ROUTES)
        assert abs(sum(scores.values()) - 1.0) < 1e-6
