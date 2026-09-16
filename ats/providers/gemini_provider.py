"""Proveedor Google Gemini (SDK oficial `google-genai`)."""

from __future__ import annotations

import os
from typing import Any, Dict, Tuple, Type

from pydantic import BaseModel

from .base import LLMError, LLMProvider, TransientLLMError


class GeminiProvider(LLMProvider):
    name = "gemini"
    label = "Gemini"
    default_model = "gemini-3.8-flash"
    env_key = "GEMINI_API_KEY"
    env_model = "GEMINI_MODEL"
    package = "google-genai"

    #: `thinking_level` solo llega hasta HIGH.
    EFFORT_MAP = {
        "low": "LOW",
        "medium": "MEDIUM",
        "high": "HIGH",
        "xhigh": "HIGH",
        "max": "HIGH",
    }

    @classmethod
    def installed(cls) -> bool:
        import importlib.util

        return importlib.util.find_spec("google.genai") is not None

    def has_credentials(self) -> bool:
        # El SDK también acepta GOOGLE_API_KEY.
        return bool(self.api_key or os.environ.get("GOOGLE_API_KEY"))

    def client(self) -> Any:
        if self._client is None:
            from google import genai

            key = self.api_key or os.environ.get("GOOGLE_API_KEY", "")
            self._client = genai.Client(api_key=key)
        return self._client

    def _call(
        self, *, system: str, user: str, output_format: Type[BaseModel], effort: str
    ) -> Tuple[BaseModel, Dict[str, Any]]:
        from google.genai import types

        client = self.client()
        config: Dict[str, Any] = {
            "system_instruction": system,
            "response_mime_type": "application/json",
            "response_schema": output_format,
            # Nunca usamos function calling; desactivarlo evita el aviso que el
            # SDK emite en cada llamada a `generate_content`.
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
            "thinking_config": types.ThinkingConfig(
                thinking_level=self.EFFORT_MAP.get(effort, "MEDIUM")
            ),
        }

        try:
            response = client.models.generate_content(
                model=self.model,
                contents=user,
                config=types.GenerateContentConfig(**config),
            )
        except TypeError:
            # Un modelo sin control de pensamiento rechaza `thinking_config`.
            config.pop("thinking_config", None)
            try:
                response = client.models.generate_content(
                    model=self.model,
                    contents=user,
                    config=types.GenerateContentConfig(**config),
                )
            except Exception as exc:  # noqa: BLE001
                raise self._translate(exc) from exc
        except Exception as exc:  # noqa: BLE001
            raise self._translate(exc) from exc

        return self._read(response)

    def _read(self, response: Any) -> Tuple[BaseModel, Dict[str, Any]]:
        feedback = getattr(response, "prompt_feedback", None)
        blocked = getattr(feedback, "block_reason", None)
        if blocked:
            raise LLMError(f"Petición bloqueada por filtros de seguridad ({blocked}).")

        candidates = getattr(response, "candidates", None) or []
        if candidates:
            finish = str(getattr(candidates[0], "finish_reason", "") or "")
            if finish and finish.upper().endswith("MAX_TOKENS"):
                raise LLMError("Respuesta truncada por límite de tokens.")
            if finish and not finish.upper().endswith("STOP"):
                raise LLMError(f"Generación interrumpida ({finish}).")

        parsed = getattr(response, "parsed", None)
        if parsed is None:
            raise LLMError("La respuesta no contenía salida estructurada válida.")

        usage = getattr(response, "usage_metadata", None)
        output = int(getattr(usage, "candidates_token_count", 0) or 0)
        # Los tokens de razonamiento se facturan como salida.
        output += int(getattr(usage, "thoughts_token_count", 0) or 0)
        return parsed, {
            "model": getattr(response, "model_version", self.model),
            "input_tokens": int(getattr(usage, "prompt_token_count", 0) or 0),
            "output_tokens": output,
            "cached_tokens": int(getattr(usage, "cached_content_token_count", 0) or 0),
        }

    def _translate(self, exc: Exception) -> LLMError:
        try:
            from google.genai import errors
        except Exception:  # noqa: BLE001 - sin SDK no hay tipos que discriminar
            return LLMError(f"Error inesperado: {type(exc).__name__}: {exc}")

        if isinstance(exc, errors.ClientError):
            code = getattr(exc, "code", None)
            if code == 401 or code == 403:
                return LLMError(f"Credenciales inválidas o sin permisos ({self.env_key}).")
            if code == 404:
                return LLMError(f"Modelo no encontrado: {self.model}.")
            if code == 429:
                return TransientLLMError("Límite de rate o cuota agotada.")
            return LLMError(f"Petición rechazada ({code}): {exc}")
        if isinstance(exc, errors.ServerError):
            return TransientLLMError(f"Error del servidor de Gemini: {exc}")
        if isinstance(exc, errors.APIError):
            return LLMError(f"Error de la API de Gemini: {exc}")
        return LLMError(f"Error inesperado: {type(exc).__name__}: {exc}")
