"""Fachada de la capa LLM.

El grafo solo conoce esto: una `ProviderChain` con un método `structured()`.
Qué vendor hay detrás (Claude, OpenAI, Gemini o un endpoint compatible) y en
qué orden se intentan lo decide `ats.providers`.
"""

from __future__ import annotations

from .providers import (
    PROVIDERS,
    AllProvidersFailed,
    LLMError,
    LLMProvider,
    LLMResult,
    ProviderChain,
    ProviderUnavailable,
    TransientLLMError,
    build_chain,
    chain_order,
    inventory,
    parse_spec,
    provider_name,
)

__all__ = [
    "PROVIDERS",
    "AllProvidersFailed",
    "LLMError",
    "LLMProvider",
    "LLMResult",
    "ProviderChain",
    "ProviderUnavailable",
    "TransientLLMError",
    "build_chain",
    "chain_order",
    "inventory",
    "parse_spec",
    "provider_name",
]
