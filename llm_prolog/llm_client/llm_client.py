"""
Thin OpenRouter client used by all LLM‑backed modules.

The client exposes a small surface area tailored to this project:
- `LLMClient.generate` for free‑form text responses.
- `LLMClient.generate_json` for structured JSON responses.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

from .config import OpenRouterConfig, load_openrouter_config


ChatMessage = Dict[str, str]


@dataclass
class Conversation:
    """Simple in‑memory conversation history for context reuse."""

    system_prompt: str
    messages: List[ChatMessage] = field(default_factory=list)

    def build_messages(self, user_content: str) -> List[ChatMessage]:
        combined: List[ChatMessage] = []
        if self.system_prompt:
            combined.append({"role": "system", "content": self.system_prompt})
        combined.extend(self.messages)
        combined.append({"role": "user", "content": user_content})
        return combined

    def append_assistant(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})

    def append_user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})


class LLMClient:
    """Client wrapper for the OpenRouter Chat Completions API."""

    def __init__(self, config: Optional[OpenRouterConfig] = None) -> None:
        self.config = config or load_openrouter_config()

    def _post(
        self,
        messages: List[ChatMessage],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model or self.config.model,
            "messages": messages,
            "temperature": (
                temperature if temperature is not None else self.config.temperature
            ),
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        elif self.config.max_tokens is not None:
            payload["max_tokens"] = self.config.max_tokens

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        resp = requests.post(self.config.base_url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()

    def generate(
        self,
        system_prompt: str,
        user_content: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Single‑turn helper: send a system and user message and return the text reply.
        """
        messages: List[ChatMessage] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})

        data = self._post(messages, temperature=temperature, max_tokens=max_tokens)
        return data["choices"][0]["message"]["content"]

    def generate_json(
        self,
        system_prompt: str,
        user_content: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Any:
        """
        Ask the model to return a single JSON object and parse it.

        This relies on prompt discipline; it does not use tool calling.
        """
        json_instructions = (
            "Respond ONLY with a valid JSON object. Do not include any prose "
            "before or after the JSON. Ensure the JSON is syntactically valid."
        )
        full_system = f"{system_prompt.strip()}\n\n{json_instructions}"
        raw = self.generate(
            full_system,
            user_content,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Be defensive around stray text.
        raw_stripped = raw.strip()
        try:
            return json.loads(raw_stripped)
        except json.JSONDecodeError:
            # Try to salvage a JSON object if the model wrapped it in extra text.
            start = raw_stripped.find("{")
            end = raw_stripped.rfind("}")
            if start != -1 and end != -1 and end > start:
                candidate = raw_stripped[start : end + 1]
                return json.loads(candidate)
            raise

    def new_conversation(self, system_prompt: str) -> Conversation:
        """Create a reusable conversation handle."""
        return Conversation(system_prompt=system_prompt)

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
        reply = data["choices"][0]["message"]["content"]
        conversation.append_user(user_content)
        conversation.append_assistant(reply)
        return reply

