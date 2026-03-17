"""
Central async executor for LLM calls with concurrency limits and optional retry/backoff.

All pipeline LLM traffic should go through LLMExecutor so global concurrency
and rate-limit handling are enforced in one place.
"""

from __future__ import annotations

import asyncio
import random
from typing import Any, Optional

from .llm_client.async_llm_client import AsyncLLMClient


# Retry policy: (max_attempts, base_delay_seconds)
# None means no retries beyond the first attempt.
RetryPolicy = Optional[tuple[int, float]]


def _is_retryable_http_status(status: int) -> bool:
    """Return True for transient errors we may retry."""
    return status == 429 or (500 <= status <= 599)


async def _backoff_delay(attempt: int, base_delay: float) -> None:
    """Exponential backoff with jitter."""
    delay = base_delay * (2**attempt)
    jitter = random.uniform(0, delay * 0.2)
    await asyncio.sleep(delay + jitter)


class LLMExecutor:
    """
    Async executor that wraps AsyncLLMClient and enforces:
    - global max_in_flight concurrent LLM calls (asyncio.Semaphore)
    - optional retry with exponential backoff on 429 / 5xx
    """

    def __init__(
        self,
        client: AsyncLLMClient,
        max_in_flight: int = 8,
        retry_policy: RetryPolicy = (3, 1.0),
    ) -> None:
        self._client = client
        self._sem = asyncio.Semaphore(max_in_flight)
        self._retry_policy = retry_policy

    async def generate(
        self,
        system_prompt: str,
        user_content: str,
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Run a single generate call through the executor (concurrency + optional retry).
        """
        async with self._sem:
            return await self._generate_with_retry(
                system_prompt,
                user_content,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )

    async def _generate_with_retry(
        self,
        system_prompt: str,
        user_content: str,
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        policy = self._retry_policy
        max_attempts = policy[0] if policy else 1
        base_delay = policy[1] if policy else 0.0

        last_error: Optional[Exception] = None
        for attempt in range(max_attempts):
            try:
                return await self._client.generate(
                    system_prompt,
                    user_content,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except Exception as e:
                last_error = e
                status = getattr(e, "response", None)
                if status is not None:
                    status_code = getattr(status, "status_code", None)
                    if status_code is not None and _is_retryable_http_status(status_code):
                        if attempt < max_attempts - 1:
                            await _backoff_delay(attempt, base_delay)
                            continue
                # Non-retryable or no more attempts
                raise
        if last_error is not None:
            raise last_error
        raise RuntimeError("LLMExecutor generate: unexpected loop exit")

    async def generate_json(
        self,
        system_prompt: str,
        user_content: str,
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Any:
        """
        Run a single generate_json call through the executor (concurrency + optional retry).
        """
        async with self._sem:
            return await self._generate_json_with_retry(
                system_prompt,
                user_content,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )

    async def _generate_json_with_retry(
        self,
        system_prompt: str,
        user_content: str,
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Any:
        policy = self._retry_policy
        max_attempts = policy[0] if policy else 1
        base_delay = policy[1] if policy else 0.0

        last_error: Optional[Exception] = None
        for attempt in range(max_attempts):
            try:
                return await self._client.generate_json(
                    system_prompt,
                    user_content,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except Exception as e:
                last_error = e
                status = getattr(e, "response", None)
                if status is not None:
                    status_code = getattr(status, "status_code", None)
                    if status_code is not None and _is_retryable_http_status(status_code):
                        if attempt < max_attempts - 1:
                            await _backoff_delay(attempt, base_delay)
                            continue
                raise
        if last_error is not None:
            raise last_error
        raise RuntimeError("LLMExecutor generate_json: unexpected loop exit")
