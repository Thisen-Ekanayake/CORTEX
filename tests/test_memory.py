"""Tests for cortex/memory.py — ConversationMemory rolling buffer."""

import re

import pytest

from cortex.memory import ConversationMemory, MemoryItem


class TestMemoryItem:
    def test_fields_are_stored(self):
        item = MemoryItem(query="hello", answer="world", timestamp="12:00:00")
        assert item.query == "hello"
        assert item.answer == "world"
        assert item.timestamp == "12:00:00"


class TestConversationMemory:
    def test_starts_empty(self):
        mem = ConversationMemory()
        assert mem.history == []

    def test_add_stores_item(self):
        mem = ConversationMemory()
        mem.add("what is AI?", "AI is…")
        assert len(mem.history) == 1
        assert mem.history[0].query == "what is AI?"
        assert mem.history[0].answer == "AI is…"

    def test_add_generates_timestamp(self):
        mem = ConversationMemory()
        mem.add("q", "a")
        ts = mem.history[0].timestamp
        assert re.match(r"^\d{2}:\d{2}:\d{2}$", ts), f"bad timestamp: {ts!r}"

    def test_rolling_buffer_drops_oldest(self):
        mem = ConversationMemory(max_items=3)
        for i in range(5):
            mem.add(f"q{i}", f"a{i}")
        assert len(mem.history) == 3
        # Oldest two are gone; only q2, q3, q4 remain
        queries = [item.query for item in mem.history]
        assert queries == ["q2", "q3", "q4"]

    def test_buffer_at_exact_capacity(self):
        mem = ConversationMemory(max_items=2)
        mem.add("q0", "a0")
        mem.add("q1", "a1")
        assert len(mem.history) == 2

    def test_all_queries_returns_ordered_list(self):
        mem = ConversationMemory()
        mem.add("first", "a")
        mem.add("second", "b")
        mem.add("third", "c")
        assert mem.all_queries() == ["first", "second", "third"]

    def test_all_queries_empty_when_no_history(self):
        assert ConversationMemory().all_queries() == []

    def test_custom_max_items(self):
        mem = ConversationMemory(max_items=1)
        mem.add("old", "a")
        mem.add("new", "b")
        assert len(mem.history) == 1
        assert mem.history[0].query == "new"
