"""Tests for cortex/streaming.py — StreamHandler token callback."""

from cortex.streaming import StreamHandler


class TestStreamHandler:
    def test_on_token_is_called_per_token(self):
        received: list[str] = []
        handler = StreamHandler(on_token=received.append)
        handler.on_llm_new_token("hello")
        assert received == ["hello"]

    def test_multiple_tokens_accumulate_in_order(self):
        received: list[str] = []
        handler = StreamHandler(on_token=received.append)
        for word in ["The", " ", "answer", " ", "is", " ", "42"]:
            handler.on_llm_new_token(word)
        assert "".join(received) == "The answer is 42"

    def test_empty_token_is_passed_through(self):
        received: list[str] = []
        handler = StreamHandler(on_token=received.append)
        handler.on_llm_new_token("")
        assert received == [""]

    def test_callback_exception_propagates(self):
        def boom(token: str) -> None:
            raise ValueError("bad token")

        handler = StreamHandler(on_token=boom)
        try:
            handler.on_llm_new_token("x")
            assert False, "expected ValueError"
        except ValueError as exc:
            assert "bad token" in str(exc)
