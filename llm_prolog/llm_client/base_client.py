from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

from .config import OpenRouterConfig, load_openrouter_config

ChatMessage = Dict[str, str]


@dataclass
class Conversation:
    """Simple in-memory conversation history for context reuse."""

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


class BaseLLMClient:
    """
    Shared functionality for sync and async LLM clients.

    This class encapsulates:
    - request payload and header construction
    - common message building helpers
    - response extraction and JSON parsing utilities

    Subclasses are responsible for implementing the actual HTTP POST
    behaviour (sync or async) and any lifecycle management for HTTP clients.
    """

    def __init__(self, config: Optional[OpenRouterConfig] = None) -> None:
        self.config = config or load_openrouter_config()
    
    def new_conversation(self, system_prompt: str) -> Conversation:
        """Create a reusable conversation handle."""
        return Conversation(system_prompt=system_prompt)
    
    def get_credits_usage_balance(self) -> Dict[str, Any]:
        """
        Fetch current OpenRouter credits and compute remaining balance.

        Requires a management key; non-management keys may receive 403.
        Returns a dict with keys: total_credits, total_usage, balance, raw.
        """
        headers = self._build_headers()
        resp = requests.get(self.config.credits_url, headers=headers, timeout=30)
        resp.raise_for_status()
        raw: Dict[str, Any] = resp.json()
        data = raw.get("data") or {}

        total_credits = float(data.get("total_credits", 0.0))
        total_usage = float(data.get("total_usage", 0.0))

        return {
            "total_credits": total_credits,
            "total_usage": total_usage,
            "balance": total_credits - total_usage,
            "raw": raw,
        }

    # ------------------------------------------------------------------
    # Request construction helpers
    # ------------------------------------------------------------------
    def _build_payload(
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
        return payload

    def _build_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

    def _build_single_turn_messages(
        self,
        system_prompt: str,
        user_content: str,
    ) -> List[ChatMessage]:
        messages: List[ChatMessage] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})
        return messages

    # ------------------------------------------------------------------
    # Response helpers
    # ------------------------------------------------------------------
    def _extract_content(self, data: Dict[str, Any]) -> str:
        return data["choices"][0]["message"]["content"]

    # ------------------------------------------------------------------
    # JSON-generation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _json_instructions() -> str:
        return (
            "Respond ONLY with a valid JSON object. Do not include any prose "
            "before or after the JSON. Ensure the JSON is syntactically valid."
        )

    def _build_json_system_prompt(self, system_prompt: str) -> str:
        return f"{system_prompt.strip()}\n\n{self._json_instructions()}"

    def _parse_json_response(self, raw: str) -> Any:
        raw_stripped = raw.strip()
        try:
            return json.loads(raw_stripped)
        except json.JSONDecodeError:
            start = raw_stripped.find("{")
            end = raw_stripped.rfind("}")
            if start != -1 and end != -1 and end > start:
                candidate = raw_stripped[start : end + 1]
                return json.loads(candidate)
            raise

