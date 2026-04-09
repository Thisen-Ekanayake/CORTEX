"""Tests for cortex/rl_dashboard.py — pure ASCII rendering functions."""

from unittest.mock import MagicMock

import pytest

from cortex.rl_dashboard import print_ascii_chart, print_confusion_matrix
from cortex.rl_router import RoutingFeedback


# ── print_ascii_chart ─────────────────────────────────────────────────────────

class TestPrintAsciiChart:
    def test_empty_data_prints_no_data_message(self, capsys):
        print_ascii_chart([])
        assert "No data" in capsys.readouterr().out

    def test_single_value_does_not_crash(self, capsys):
        print_ascii_chart([0.5])
        out = capsys.readouterr().out
        assert "│" in out   # chart body rendered

    def test_uniform_data_renders_full_bar(self, capsys):
        print_ascii_chart([1.0] * 20, height=3, width=20)
        out = capsys.readouterr().out
        assert "█" in out

    def test_data_longer_than_width_is_downsampled(self, capsys):
        # Should not crash when len(data) > width
        print_ascii_chart(list(range(100)), height=5, width=20)
        capsys.readouterr()   # just verifying no exception

    def test_custom_title_appears_in_output(self, capsys):
        print_ascii_chart([0.5, 0.7], title="My Chart")
        assert "My Chart" in capsys.readouterr().out


# ── print_confusion_matrix ────────────────────────────────────────────────────

def _make_feedback(predicted: str, actual: str, correct: bool) -> RoutingFeedback:
    return RoutingFeedback(
        query="test",
        predicted_route=predicted,
        actual_route=actual,
        confidence_scores={predicted: 0.8},
        correct=correct,
        timestamp="12:00:00",
        reward=1.0 if correct else -1.0,
    )


class TestPrintConfusionMatrix:
    def test_empty_feedbacks_prints_no_data_message(self, capsys):
        print_confusion_matrix([])
        assert "No feedback" in capsys.readouterr().out

    def test_correct_prediction_shown_on_diagonal(self, capsys):
        feedbacks = [_make_feedback("chat", "chat", correct=True)]
        print_confusion_matrix(feedbacks)
        out = capsys.readouterr().out
        assert "CHAT" in out
        assert "1" in out

    def test_incorrect_prediction_increments_off_diagonal(self, capsys):
        feedbacks = [
            _make_feedback("chat", "meta", correct=False),
            _make_feedback("chat", "meta", correct=False),
        ]
        print_confusion_matrix(feedbacks)
        out = capsys.readouterr().out
        assert "CHAT" in out
        assert "META" in out
