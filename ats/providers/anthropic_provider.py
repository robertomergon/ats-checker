"""Proveedor Claude (SDK oficial `anthropic`)."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Type

from pydantic import BaseModel

from .base import LLMError, LLMProvider, TransientLLMError

#: Beta que habilita `fallbacks="default"`: si un clasificador de seguridad
#: rechaza la petición, Anthropic la reintenta server-side en otro modelo en
#: lugar de devolver `stop_reason="refusal"`. Es un fallback *dentro* de
#: Anthropic, complementario a la cadena multi-proveedor.
REFUSAL_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicProvider(LLMProvider):
    name = "anthropic"
    label = "Claude"
    default_model = "claude-opus-5"
    env_key = "ANTHROPIC_API_KEY"
    env_model = "ANTHROPIC_MODEL"
    env_base_url = "ANTHROPIC_BASE_URL"
    package = "anthropic"
    #: Tarifas de Claude Opus 5 (USD por millón de tokens).
    price_per_mtok = (5.0, 25.0)

    #: Claude admite los mismos niveles que usa la app.
    EFFORT_MAP = {e: e for e in ("low", "medium", "high", "xhigh", "max")}

    def has_credentials(self) -> bool:
        import os

        # El SDK también resuelve credenciales por perfil o token de entorno.
        return bool(self.api_key or os.environ.get("ANTHROPIC_AUTH_TOKEN"))

    def client(self) -> Any:
        if self._client is None:
            import anthropic

            kwargs: Dict[str, Any] = {}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = anthropic.Anthropic(**kwargs)
        return self._client

    def _call(
        self, *, system: str, user: str, output_format: Type[BaseModel], effort: str
    ) -> Tuple[BaseModel, Dict[str, Any]]:
        client = self.client()
        base: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": 16_000,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "output_format": output_format,
        }
        mapped_effort = self.EFFORT_MAP.get(effort, "medium")

        # Degradación ordenada: si el SDK instalado no conoce un parámetro
        # (TypeError) se reintenta con un conjunto más reducido.
        attempts = [
            (
                "beta",
                dict(
                    base,
                    betas=[REFUSAL_FALLBACK_BETA],
                    fallbacks="default",
                    thinking={"type": "adaptive"},
                    output_config={"effort": mapped_effort},
                ),
            ),
            (
                "beta",
                dict(
                    base,
                    betas=[REFUSAL_FALLBACK_BETA],
                    fallbacks="default",
                    thinking={"type": "adaptive"},
                ),
            ),
            (
                "plain",
                dict(base, thinking={"type": "adaptive"}, output_config={"effort": mapped_effort}),
            ),
            ("plain", dict(base, thinking={"type": "adaptive"})),
            ("plain", dict(base)),
        ]

        last_signature_error: Optional[Exception] = None
        for surface, kwargs in attempts:
            namespace = client.beta.messages if surface == "beta" else client.messages
            try:
                response = namespace.parse(**kwargs)
            except (TypeError, AttributeError) as exc:
                last_signature_error = exc
                continue
            except Exception as exc:  # noqa: BLE001 - se traduce a LLMError
                raise self._translate(exc) from exc
            return self._read(response)

        raise LLMError(
            "El SDK `anthropic` instalado no acepta los parámetros necesarios "
            f"({last_signature_error}). Actualiza con: uv pip install -U anthropic"
        )

    # --- Lectura de la respuesta -----------------------------------------

    def _read(self, response: Any) -> Tuple[BaseModel, Dict[str, Any]]:
        # `stop_reason` se comprueba antes del contenido: en un rechazo no hay
        # salida estructurada que parsear.
        if getattr(response, "stop_reason", None) == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) or "sin categoría"
            raise LLMError(f"Petición rechazada por políticas de seguridad ({category}).")

        parsed = getattr(response, "parsed_output", None)
        if parsed is None:
            raise LLMError("La respuesta no contenía salida estructurada válida.")

        usage = getattr(response, "usage", None)
        return parsed, {
            "model": getattr(response, "model", self.model),
            "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
            "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
            "cached_tokens": int(getattr(usage, "cache_read_input_tokens", 0) or 0),
        }

    def _translate(self, exc: Exception) -> LLMError:
        """De lo específico a lo general, como recomienda el SDK."""
        import anthropic

        if isinstance(exc, anthropic.AuthenticationError):
            return LLMError("Credenciales inválidas (ANTHROPIC_API_KEY).")
        if isinstance(exc, anthropic.PermissionDeniedError):
            return LLMError("La API key no tiene permiso para usar este modelo.")
        if isinstance(exc, anthropic.NotFoundError):
            return LLMError(f"Modelo no disponible para esta cuenta: {self.model}.")
        if isinstance(exc, anthropic.RateLimitError):
            return TransientLLMError("Límite de rate alcanzado.")
        if isinstance(exc, anthropic.BadRequestError):
            return LLMError(f"Petición rechazada por la API: {exc}")
        if isinstance(exc, anthropic.APITimeoutError):
            return TransientLLMError("Tiempo de espera agotado.")
        if isinstance(exc, anthropic.APIConnectionError):
            return TransientLLMError("No se pudo conectar con la API de Anthropic.")
        if isinstance(exc, anthropic.APIStatusError):
            status = getattr(exc, "status_code", 0) or 0
            message = f"Error HTTP {status or '?'}: {exc}"
            return TransientLLMError(message) if status >= 500 else LLMError(message)
        return LLMError(f"Error inesperado: {type(exc).__name__}: {exc}")
