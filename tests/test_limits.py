"""Tests for the in-process question validation and global limiter."""
from __future__ import annotations

import pytest

from src import config, limits


class TestValidateQuestion:
    def test_accepts_a_normal_question(self):
        assert limits.validate_question("What is the exit load?") is None

    def test_rejects_empty_and_whitespace_only_questions(self):
        assert limits.validate_question("") == "Please enter a question."
        assert limits.validate_question("   \n\t") == "Please enter a question."

    def test_rejects_a_question_over_the_limit(self):
        too_long = "x" * (config.MAX_QUESTION_CHARS + 1)
        assert "under" in limits.validate_question(too_long)

    def test_accepts_exactly_the_limit(self):
        at_limit = "x" * config.MAX_QUESTION_CHARS
        assert limits.validate_question(at_limit) is None

    def test_uses_the_configured_maximum(self):
        assert limits.validate_question("abcd", max_chars=3)


class TestGlobalQuestionLimiter:
    def test_allows_requests_up_to_the_window_limit(self):
        now = [0.0]
        limiter = limits.GlobalQuestionLimiter(
            max_requests=2,
            window_seconds=10,
            max_concurrent=2,
            clock=lambda: now[0],
        )
        assert limiter.try_acquire().allowed
        limiter.release()
        assert limiter.try_acquire().allowed
        limiter.release()

        denied = limiter.try_acquire()
        assert denied.allowed is False
        assert denied.reason == "rate_limited"
        assert denied.retry_after_s == 10

    def test_window_slides_after_the_oldest_request_expires(self):
        now = [0.0]
        limiter = limits.GlobalQuestionLimiter(
            max_requests=1,
            window_seconds=10,
            max_concurrent=1,
            clock=lambda: now[0],
        )
        assert limiter.try_acquire().allowed
        limiter.release()
        assert limiter.try_acquire().reason == "rate_limited"

        now[0] = 10.1
        assert limiter.try_acquire().allowed
        limiter.release()

    def test_retry_after_is_at_least_one_second(self):
        now = [9.9]
        limiter = limits.GlobalQuestionLimiter(
            max_requests=1,
            window_seconds=10,
            max_concurrent=1,
            clock=lambda: now[0],
        )
        now[0] = 0.0
        assert limiter.try_acquire().allowed
        limiter.release()
        now[0] = 9.9
        assert limiter.try_acquire().retry_after_s == 1

    def test_denies_when_all_concurrency_slots_are_in_use(self):
        limiter = limits.GlobalQuestionLimiter(
            max_requests=10,
            window_seconds=60,
            max_concurrent=1,
        )
        assert limiter.try_acquire().allowed
        denied = limiter.try_acquire()
        assert denied.allowed is False
        assert denied.reason == "busy"

        limiter.release()
        assert limiter.try_acquire().allowed
        limiter.release()

    def test_get_global_limiter_is_a_singleton(self):
        assert limits.get_global_limiter() is limits.get_global_limiter()

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"max_requests": 0},
            {"max_requests": 1, "window_seconds": 0},
            {"max_requests": 1, "max_concurrent": 0},
        ],
    )
    def test_rejects_invalid_configuration(self, kwargs):
        with pytest.raises(ValueError):
            limits.GlobalQuestionLimiter(**kwargs)
