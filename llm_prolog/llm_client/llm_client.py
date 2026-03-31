"""
Thin OpenRouter client used by all LLM‑backed modules.

The client exposes a small surface area tailored to this project:
- `LLMClient.generate` for free‑form text responses.
- `LLMClient.generate_json` for structured JSON responses.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import requests

from .base_client import BaseLLMClient, ChatMessage, Conversation
from .config import OpenRouterConfig, load_openrouter_config


class LLMClient(BaseLLMClient):
    """Client wrapper for the OpenRouter Chat Completions API."""

    def __init__(self, config: Optional[OpenRouterConfig] = None) -> None:
        super().__init__(config=config)

    def _post(
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

        resp = requests.post(
            self.config.base_url,
            headers=headers,
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        body = resp.json()
        self._accumulate_usage_from_response(body)
        return body

    def generate(
        self,
        system_prompt: str,
        user_content: str,
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Single‑turn helper: send a system and user message and return the text reply.
        """
        messages = self._build_single_turn_messages(system_prompt, user_content)

        data = self._post(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return self._extract_content(data)

    def generate_json(
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
        Ask the model to return a single JSON object and parse it.

        This relies on prompt discipline; it does not use tool calling.
        """
        full_system = self._build_json_system_prompt(system_prompt)
        raw = self.generate(
            full_system,
            user_content,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return self._parse_json_response_with_optional_raw(raw, return_raw=return_raw)

    def continue_conversation(
        self,
        conversation: Conversation,
        user_content: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Append a user message to an existing conversation, send it, and record
        the assistant reply back into the conversation history.
        """
        messages = conversation.build_messages(user_content)
        data = self._post(messages, temperature=temperature, max_tokens=max_tokens)
        reply = self._extract_content(data)
        conversation.append_user(user_content)
        conversation.append_assistant(reply)
        return reply

