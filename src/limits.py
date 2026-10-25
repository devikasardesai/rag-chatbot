"""In-process abuse controls for the Streamlit UI.

These limits are deliberately best-effort. They are shared by every session in one
process, reset when the process restarts, and are not shared between workers or hosted
instances. A provider-side spending limit remains the only hard cost control.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

from src import config


@dataclass(frozen=True)
class LimitDecision:
    """Outcome of a request for one global question slot."""

    allowed: bool
    reason: str | None = None
    message: str = ""
    retry_after_s: int = 0


def validate_question(question: str, max_chars: int = config.MAX_QUESTION_CHARS) -> str | None:
    """Return a user-facing validation message, or ``None`` when the question is valid."""
    text = str(question or "").strip()
    if not text:
        return "Please enter a question."
    if len(text) > max_chars:
        return f"Please keep your question under {max_chars} characters."
    return None


class GlobalQuestionLimiter:
    """Process-wide request-rate and concurrency limiter.

    ``try_acquire`` consumes one request from the rolling window and, when allowed,
    reserves one concurrency slot. Every successful acquisition must be paired with
    :meth:`release` in a ``finally`` block.
    """

    def __init__(
        self,
        max_requests: int = config.MAX_GLOBAL_QUESTIONS_PER_WINDOW,
        window_seconds: float = config.GLOBAL_QUESTION_WINDOW_SECONDS,
        max_concurrent: int = config.MAX_CONCURRENT_QUESTIONS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_requests < 1:
            raise ValueError("max_requests must be at least 1")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be at least 1")
        self._max_requests = int(max_requests)
        self._window_seconds = float(window_seconds)
        self._clock = clock
        self._timestamps: deque[float] = deque()
        self._lock = threading.Lock()
        self._slots = threading.BoundedSemaphore(int(max_concurrent))

    def try_acquire(self) -> LimitDecision:
        """Consume one request and reserve a concurrency slot when possible."""
        now = self._clock()
        with self._lock:
            cutoff = now - self._window_seconds
            while self._timestamps and self._timestamps[0] <= cutoff:
                self._timestamps.popleft()
            if len(self._timestamps) >= self._max_requests:
                retry_after = max(
                    1, int(round(self._timestamps[0] + self._window_seconds - now))
                )
                return LimitDecision(
                    allowed=False,
                    reason="rate_limited",
                    message=(
                        "The assistant has reached its temporary question limit. "
                        f"Please try again in about {retry_after} seconds."
                    ),
                    retry_after_s=retry_after,
                )
            self._timestamps.append(now)

        if not self._slots.acquire(blocking=False):
            return LimitDecision(
                allowed=False,
                reason="busy",
                message=(
                    "The assistant is handling too many questions right now. "
                    "Please try again in a moment."
                ),
            )
        return LimitDecision(allowed=True)

    def release(self) -> None:
        """Release a slot reserved by :meth:`try_acquire`."""
        self._slots.release()


_global_limiter: GlobalQuestionLimiter | None = None
_global_limiter_lock = threading.Lock()


def get_global_limiter() -> GlobalQuestionLimiter:
    """Return the process-wide limiter used by the Streamlit app."""
    global _global_limiter
    if _global_limiter is None:
        with _global_limiter_lock:
            if _global_limiter is None:
                _global_limiter = GlobalQuestionLimiter()
    return _global_limiter
