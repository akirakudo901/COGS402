"""
Async OpenRouter client for concurrent LLM calls.

Mirrors LLMClient (generate, generate_json) using httpx.AsyncClient for
non-blocking I/O. Use a single client instance per run for connection pooling.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import httpx

from .base_client import BaseLLMClient, ChatMessage
from .config import OpenRouterConfig



DEFAULT_TIMEOUT = 60.0


class AsyncLLMClient(BaseLLMClient):
    """Async client for the OpenRouter Chat Completions API using httpx."""

    def __init__(
        self,
        config: Optional[OpenRouterConfig] = None,
        *,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        super().__init__(config=config)
        self._timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def aclose(self) -> None:
        """Close the underlying HTTP client. Call when done with the client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "AsyncLLMClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.aclose()

    async def _post(
        self,
        messages: List[ChatMessage],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        payload = self._build_payload(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        headers = self._build_headers()

        client = self._get_client()
        resp = await client.post(
            self.config.base_url,
            headers=headers,
            json=payload,
        )
        resp.raise_for_status()
        body = resp.json()
        self._accumulate_usage_from_response(body)
        return body

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
        Single-turn async helper: send system and user message, return text reply.
        """
        messages = self._build_single_turn_messages(system_prompt, user_content)

        data = await self._post(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return self._extract_content(data)

    async def generate_json(
        self,
        system_prompt: str,
        user_content: str,
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        return_raw: bool = False,
    ) -> Union[Any, Tuple[Any, str]]:
        """
        Ask the model to return a single JSON object and parse it (async).
        """
        full_system = self._build_json_system_prompt(system_prompt)
        raw = await self.generate(
            full_system,
            user_content,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return self._parse_json_response_with_optional_raw(raw, return_raw=return_raw)
