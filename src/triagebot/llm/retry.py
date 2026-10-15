"""Retries with exponential backoff, for errors that go away if you wait."""

from __future__ import annotations

import random
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from triagebot.llm.base import (
    UNAVAILABLE_STATUSES,
    LLMClient,
    LLMError,
    LLMResponse,
    Message,
    ToolSpec,
)

T = TypeVar("T")
RETRY_DELAY = re.compile(r'"retryDelay":\s*"(\d+(?:\.\d+)?)s"')


def http_error(status: int, text: str, headers: dict[str, str] | None = None) -> LLMError:
    """Turn an error response into an LLMError that knows whether a retry can help."""
    retry_after = None
    header = (headers or {}).get("retry-after", "")
    if header.isdigit():
        retry_after = float(header)
    elif match := RETRY_DELAY.search(text):
        retry_after = float(match.group(1))
    if status == 429 and "PerDay" in text:
        # Gemini names the exhausted quota; a daily one won't come back by waiting minutes
        return LLMError(f"HTTP 429: daily quota used up for this model. {text[:200]}", status)
    return LLMError(f"HTTP {status}: {text[:300]}", status,
                    retryable=status in UNAVAILABLE_STATUSES, retry_after=retry_after)


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 3
    base_s: float = 2.0  # waits about 2, 4, 8 seconds
    max_wait_s: float = 60.0  # a longer wait is not worth it; give up (and maybe fall back)


def with_retries(
    call: Callable[[], T],
    policy: RetryPolicy,
    say: Callable[[str], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    rand: Callable[[], float] = random.random,
) -> T:
    attempt = 0
    while True:
        try:
            return call()
        except LLMError as exc:
            if not exc.retryable or attempt >= policy.max_retries:
                raise
            # Jitter (0.5x to 1.5x) stops many clients from retrying at the same moment
            wait = exc.retry_after or policy.base_s * 2 ** attempt * (0.5 + rand())
            if wait > policy.max_wait_s:
                raise
            attempt += 1
            if say:
                say(f"      {' '.join(str(exc).split())[:80]}... retry {attempt}/{policy.max_retries} "
                    f"in {wait:.0f} s")
            sleep(wait)


class RetryingClient:
    """Wraps any LLMClient so that every chat call is retried."""

    def __init__(self, inner: LLMClient, policy: RetryPolicy,
                 say: Callable[[str], None] | None = None,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.inner = inner
        self.model = inner.model
        self.policy = policy
        self.say = say
        self.sleep = sleep

    def chat(self, messages: list[Message], tools: list[ToolSpec] | None = None) -> LLMResponse:
        return with_retries(lambda: self.inner.chat(messages, tools), self.policy, self.say,
                            self.sleep)
