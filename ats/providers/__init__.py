"""Registro de proveedores LLM y construcción de la cadena de fallback.

Configuración por `.env` (ver `.env.example`):

    LLM_PROVIDER=anthropic          # proveedor principal
    LLM_FALLBACKS=gemini,openai     # orden de respaldo; si se omite, se usan
                                    # automáticamente los demás con credenciales

Cada entrada admite `proveedor:modelo` para fijar el modelo de ese eslabón, lo
que permite encadenar dos modelos del mismo proveedor:

    LLM_PROVIDER=gemini                          # usa GEMINI_MODEL
    LLM_FALLBACKS=gemini:gemini-3.6-flash        # respaldo al modelo anterior
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

from .anthropic_provider import AnthropicProvider
from .base import (
    AllProvidersFailed,
    LLMError,
    LLMProvider,
    LLMResult,
    ProviderChain,
    ProviderUnavailable,
    TransientLLMError,
)
from .gemini_provider import GeminiProvider
from .openai_provider import OpenAICompatibleProvider, OpenAIProvider

#: Orden canónico: define también el orden de fallback automático.
PROVIDERS: Dict[str, Type[LLMProvider]] = {
    AnthropicProvider.name: AnthropicProvider,
    OpenAIProvider.name: OpenAIProvider,
    GeminiProvider.name: GeminiProvider,
    OpenAICompatibleProvider.name: OpenAICompatibleProvider,
}

DEFAULT_PRIMARY = "anthropic"

__all__ = [
    "PROVIDERS",
    "AllProvidersFailed",
    "AnthropicProvider",
    "GeminiProvider",
    "LLMError",
    "LLMProvider",
    "LLMResult",
    "OpenAICompatibleProvider",
    "OpenAIProvider",
    "ProviderChain",
    "ProviderUnavailable",
    "TransientLLMError",
    "build_chain",
    "chain_order",
    "inventory",
    "parse_spec",
    "provider_name",
]


def _split(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_spec(spec: str) -> Tuple[str, Optional[str]]:
    """Convierte `"gemini:gemini-3.6-flash"` en `("gemini", "gemini-3.6-flash")`.

    Sin `:modelo`, el modelo lo decide el entorno o el valor por defecto del
    proveedor. Se admite `:` dentro del nombre del modelo (`openai:ft:gpt-x`).
    """
    name, _, model = str(spec).strip().partition(":")
    return name.strip().lower(), (model.strip() or None)


def provider_name(spec: str) -> str:
    return parse_spec(spec)[0]


def chain_order(
    primary: Optional[str] = None, fallbacks: Optional[Sequence[str]] = None
) -> List[str]:
    """Orden de proveedores a intentar, sin filtrar por disponibilidad.

    Si no se indican respaldos ni en argumentos ni en `LLM_FALLBACKS`, se usan
    todos los demás proveedores conocidos en su orden canónico: así hay
    fallback sin necesidad de configurar nada.
    """
    primary = (primary or os.environ.get("LLM_PROVIDER") or DEFAULT_PRIMARY).strip()
    if provider_name(primary) not in PROVIDERS:
        primary = DEFAULT_PRIMARY

    if fallbacks is None:
        env_value = os.environ.get("LLM_FALLBACKS")
        specs = _split(env_value) if env_value is not None else [
            name for name in PROVIDERS if name != provider_name(primary)
        ]
    else:
        specs = [str(f).strip() for f in fallbacks]

    # La deduplicación es por entrada completa, no por proveedor: así conviven
    # dos eslabones del mismo vendor con modelos distintos.
    order = [primary]
    for spec in specs:
        if provider_name(spec) in PROVIDERS and spec not in order:
            order.append(spec)
    return order


def build_chain(
    primary: Optional[str] = None,
    fallbacks: Optional[Sequence[str]] = None,
    *,
    api_keys: Optional[Dict[str, str]] = None,
    models: Optional[Dict[str, str]] = None,
    base_urls: Optional[Dict[str, str]] = None,
) -> ProviderChain:
    """Instancia la cadena, descartando los proveedores que no se pueden usar.

    `api_keys`, `models` y `base_urls` sobrescriben por proveedor lo que venga
    del entorno (la barra lateral los usa para valores tecleados a mano).
    """
    api_keys = api_keys or {}
    models = models or {}
    base_urls = base_urls or {}
    usable: List[LLMProvider] = []

    for spec in chain_order(primary, fallbacks):
        name, pinned_model = parse_spec(spec)
        provider = PROVIDERS[name](
            model=pinned_model or models.get(name) or None,
            api_key=api_keys.get(name) or None,
            base_url=base_urls.get(name) or None,
        )
        available, _ = provider.availability()
        if available:
            usable.append(provider)

    return ProviderChain(usable)


def inventory(
    api_keys: Optional[Dict[str, str]] = None,
    models: Optional[Dict[str, str]] = None,
    base_urls: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Estado de cada proveedor conocido, para mostrarlo en la UI."""
    api_keys = api_keys or {}
    models = models or {}
    base_urls = base_urls or {}
    rows: List[Dict[str, Any]] = []

    for name, cls in PROVIDERS.items():
        provider = cls(
            model=models.get(name) or None,
            api_key=api_keys.get(name) or None,
            base_url=base_urls.get(name) or None,
        )
        available, reason = provider.availability()
        rows.append(
            {
                "name": name,
                "label": provider.label,
                "model": provider.model or "",
                "available": available,
                "reason": reason,
                "installed": provider.installed(),
                "env_key": provider.env_key,
                "env_model": provider.env_model,
                "env_base_url": provider.env_base_url,
                "needs_base_url": bool(provider.env_base_url) and name == "custom",
                "prices": provider.prices(),
            }
        )
    return rows
