from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

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
        self._init_usage_stats()

    def _init_usage_stats(self) -> None:
        self._usage_totals: Dict[str, Any] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "n_requests": 0,
            "cost_usd": 0.0,
            "reasoning_tokens": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
        }

    def reset_usage_stats(self) -> None:
        """Reset counters for a new evaluation run (sync or async client)."""
        self._init_usage_stats()

    def get_usage_stats(self) -> Dict[str, Any]:
        """Cumulative usage from API responses (OpenRouter-style `usage` + optional cost)."""
        return dict(self._usage_totals)

    def _accumulate_usage_from_response(self, data: Dict[str, Any]) -> None:
        """Parse one chat-completions JSON body; safe no-op if `usage` is absent."""
        self._usage_totals["n_requests"] += 1
        usage = data.get("usage")
        if isinstance(usage, dict):
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                val = usage.get(key)
                if isinstance(val, int):
                    self._usage_totals[key] += val
            comp_det = usage.get("completion_tokens_details")
            if isinstance(comp_det, dict):
                rt = comp_det.get("reasoning_tokens")
                if isinstance(rt, int):
                    self._usage_totals["reasoning_tokens"] += rt
            prompt_det = usage.get("prompt_tokens_details")
            if isinstance(prompt_det, dict):
                for key in ("cached_tokens", "cache_write_tokens"):
                    v = prompt_det.get(key)
                    if isinstance(v, int):
                        self._usage_totals[key] += v
        cost_val = None
        if isinstance(usage, dict):
            cost_val = usage.get("cost")
            if cost_val is None:
                cost_val = usage.get("total_cost")
        if cost_val is None:
            cost_val = data.get("cost")
        if cost_val is None:
            cost_val = data.get("total_cost")
        if isinstance(cost_val, (int, float)):
            self._usage_totals["cost_usd"] = float(self._usage_totals["cost_usd"]) + float(cost_val)

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

    def _parse_json_response_with_optional_raw(
        self,
        raw: str,
        *,
        return_raw: bool = False,
    ) -> Union[Any, Tuple[Any, str]]:
        parsed = self._parse_json_response(raw)
        if return_raw:
            return parsed, raw
        return parsed

