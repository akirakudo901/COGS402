"""
Configuration for the LLM‑Prolog pipeline.

This module centralises the OpenRouter configuration so the rest of the
codebase does not need to worry about environment variables or URLs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv # for loading API keys from .env file

OPENROUTER_BASE_URL_DEFAULT = "https://openrouter.ai/api/v1/chat/completions"


@dataclass(frozen=True)
class OpenRouterConfig:
    """Configuration required to talk to the OpenRouter chat completions API."""

    api_key: str
    model: str = "openai/gpt-4.1-mini"
    base_url: str = OPENROUTER_BASE_URL_DEFAULT
    temperature: float = 0.2
    max_tokens: int | None = None


def load_openrouter_config(
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> OpenRouterConfig:
    """
    Load OpenRouter configuration from environment variables with sensible defaults.

    Expected environment variables:
    - OPENROUTER_API_KEY: required
    - OPENROUTER_MODEL: optional override for the default model
    """

    # load API key & optionally the model via dotenv
    load_dotenv()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY environment variable is not set. "
            "Set it to your OpenRouter API key before running the pipeline."
        )

    env_model = os.getenv("OPENROUTER_MODEL")

    return OpenRouterConfig(
        api_key=api_key,
        model=model or env_model or OpenRouterConfig.model,
        base_url=OPENROUTER_BASE_URL_DEFAULT,
        temperature=temperature if temperature is not None else OpenRouterConfig.temperature,
        max_tokens=max_tokens,
    )

